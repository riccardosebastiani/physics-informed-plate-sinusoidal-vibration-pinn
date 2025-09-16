import torch
from tqdm.autonotebook import tqdm
import time


def train(model, train_dataloader, epochs, n_step, lr, steps_til_summary, loss_fn,
          history_loss, history_lambda, metric, metric_lam, max_epochs_without_improvement, clip_grad=False,
          use_lbfgs=False):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=0.1,
        patience=30,
        min_lr=1e-16,
        eps=1e-16
    )

    total_steps = 0
    best_loss = float('inf')
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            metric.reset_state()
            metric_lam.reset_state()
            for i in range(n_step):
                for step, model_input in enumerate(train_dataloader):
                    start_time = time.time()

                    model_input = {key: value for key, value in model_input.items()}

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, model_input)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            train_loss.backward()
                            return train_loss

                        optim.step(closure)

                    model_output = model(model_input)
                    losses = loss_fn.call(model_output, model_input)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss

                    train_losses.append(train_loss.item())


                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()
                        if metric_lam is None:
                            metric.update_state(model_input, model_output)
                            metric_result = metric.result()
                        else:
                            metric.update_state(model_input, model_output, losses)
                            metric_result = metric.result()
                            metric_lam.update_state(model_input, model_output)
                            lambda_results = metric_lam.result()

                    if not total_steps % steps_til_summary:
                        current_lr = optim.param_groups[0]['lr']
                        l_u_met = metric_result['L_f'] + metric_result['L_b0'] + metric_result['L_b2'] + metric_result['L_t']
                        tqdm.write("Epoch %d, Total loss %0.3e, L_f %0.3e, L_b0 %0.3e, L_b2 %0.3e, L_u %0.3e, "
                                   "L_t %0.3e,"
                                   "iteration time %0.6f, Lam_f = %0.3f, Lam_b0 = %0.3f, Lam_b2 = %0.3f, lr: %.3e"
                                   % (epoch, l_u_met, metric_result['L_f'], metric_result['L_b0'], metric_result['L_b2'],
                                      metric_result['L_u'], metric_result['L_t'], time.time() - start_time, lambda_results['L_f'],
                                      lambda_results['L_b0'], lambda_results['L_b2'], current_lr))
                    total_steps += 1

            lr_scheduler.step(train_loss)
            try:
                history_loss['L_f'].append(metric_result['L_f'])
                history_loss['L_b0'].append(metric_result['L_b0'])
                history_loss['L_b2'].append(metric_result['L_b2'])
                history_loss['L_u'].append(metric_result['L_u'])
                history_loss['L_t'].append(metric_result['L_t'])
            except TypeError:
                print(f"Error in epoch {epoch}: metric_result = {metric_result}")

            if metric_lam is not None:
                try:
                    history_lambda['L_f_lambda'].append(lambda_results['L_f'])
                    history_lambda['L_b0_lambda'].append(lambda_results['L_b0'])
                    history_lambda['L_b2_lambda'].append(lambda_results['L_b2'])
                    history_lambda['L_t_lambda'].append(lambda_results['L_t'])

                except TypeError:
                    print(f"Error in epoch {epoch}: metric_result = {metric_result}")
            if train_loss < best_loss:
                best_loss = train_loss
                current_epochs_without_improvement = 0
                best_model_weights = model.state_dict()
                best_model_epoch = epoch
            else:
                current_epochs_without_improvement += 1

            if current_epochs_without_improvement >= max_epochs_without_improvement and train_loss < 10e-7:
                print(f'Early stopping after {epoch} epochs without improvement.')
                model.load_state_dict(best_model_weights)
                print(f'Model weights loaded from epoch {best_model_epoch}.')
                break
            pbar.update(1)


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
