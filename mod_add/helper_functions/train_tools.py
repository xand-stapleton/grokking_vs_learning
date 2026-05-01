import copy

import torch


def linear_decomposition(model, criterion, dataloader, device):
    with torch.no_grad():
        copied_model = copy.deepcopy(model)
        test_loss_full = 0.0
        test_loss_linear = 0.0
        test_loss_non_linear = 0.0
        linear_norms = 0.0
        non_linear_norms = 0.0
        diff_norms = 0.0
        for batch in dataloader:
            X_test, y_test = batch
            y_pred_full = copied_model(X_test).to(device)

            non_linear_y_pred = copied_model(X_test).to(device) + copied_model(
                -X_test
            ).to(device)
            linear_y_pred = y_pred_full - non_linear_y_pred

            float_y = y_test.float().clone().to(device)

            # X_train,y_train=batch
            # optimizer=model.optimizer

            # optimizer.zero_grad()

            # y_pred=model(X_train).to(device)

            # #loss = criterion(y_pred, y_train.to(device))

            # y_train=y_train.float().to(device)
            # loss = criterion(y_pred, y_train.to(device))
            # print(f'calculated loss correctly: {loss.item()}')

            # print(f'criterion is {criterion}')
            # print(f'shape y pred full {y_pred_full.shape}')
            # print(f'shape float y {float_y.shape}')
            # print(f'nonzeros in y pred full {torch.sum(y_pred_full!=0)}')
            # print(f'type y pred full {y_pred_full.dtype}')
            # print(f'type float y {float_y.dtype}')
            # print(f'nonzeros in float y {torch.sum(float_y!=0)}')
            # exit()

            full_loss = criterion(y_pred_full, float_y)
            test_loss_full += full_loss.item()

            non_linear_loss = criterion(non_linear_y_pred, float_y)
            test_loss_non_linear += non_linear_loss.item()

            linear_loss = criterion(linear_y_pred, float_y)
            test_loss_linear += linear_loss.item()

            linear_norm = torch.sqrt(torch.sum(linear_y_pred**2)).item()
            linear_norms += linear_norm

            non_linear_norm = torch.sqrt(torch.sum(non_linear_y_pred**2)).item()
            non_linear_norms += non_linear_norm

            diff_norm = torch.sqrt(
                torch.sum((linear_y_pred - non_linear_y_pred) ** 2)
            ).item()
            diff_norms += diff_norm
    return (
        test_loss_full,
        test_loss_linear,
        test_loss_non_linear,
        linear_norms,
        non_linear_norms,
        diff_norms,
    )


def flatten_parameters(model):
    parameters = [param for param in model.parameters() if param.requires_grad]
    flat_parameters = torch.cat([param.view(-1) for param in parameters])
    return flat_parameters


def check_weights_update(initial_weights, updated_weights):
    return not torch.equal(initial_weights, updated_weights)


def get_samples(model, weight_indices, bias_indices):
    weight_tensors = [
        weight for weight in model.parameters() if weight.dim() > 1
    ]
    bias_tensors = [bias for bias in model.parameters() if bias.dim() == 1]
    with torch.no_grad():
        sampled_weights = torch.stack(
            [
                weight[tuple(indices.T)]
                for weight, indices in zip(weight_tensors, weight_indices)
            ],
            dim=0,
        )
        sampled_biases = torch.stack(
            [
                bias[tuple(indices.T)]
                for bias, indices in zip(bias_tensors, bias_indices)
            ],
            dim=0,
        )

    return sampled_weights, sampled_biases


def split_weights(
    back_steps_model, front_steps_model, weight_keys, split_into=10
):
    model_diff = copy.deepcopy(back_steps_model)

    with torch.no_grad():
        differences = []
        front_abs_values = []
        back_abs_values = []

        for param_front, param_back, param_diff in zip(
            front_steps_model.parameters(),
            back_steps_model.parameters(),
            model_diff.parameters(),
        ):
            param_diff.copy_(param_front - param_back)
            differences.append(param_diff.view(-1))  # Flatten
            front_abs_values.append(
                param_front.abs().view(-1)
            )  # Flatten and take absolute value of front steps
            back_abs_values.append(param_back.abs().view(-1))
    # Concatenate all parameters into single tensors
    all_differences = torch.cat(differences)
    all_front_abs_values = torch.cat(front_abs_values)
    all_back_abs_values = torch.cat(back_abs_values)

    # Sort the absolute values of front steps and get indices
    sorted_abs_front_values, indices = torch.sort(all_front_abs_values)
    sorted_abs_back_values, indices = torch.sort(all_back_abs_values)

    # Use the sorted indices to order the differences accordingly
    sorted_differences = all_differences[indices]

    ratio_differences = all_differences[indices] / (
        all_back_abs_values[indices] + 1e-8
    )
    # Calculate the mean of each decile based on the sorted absolute front values
    deciles = []
    num_elements = sorted_abs_front_values.numel()
    decile_size = num_elements // split_into

    for i in range(split_into):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < split_into - 1 else num_elements
        # decile_mean = sorted_differences[start_idx:end_idx].abs().mean()
        decile_mean = ratio_differences[start_idx:end_idx].mean()  # .abs()?
        deciles.append(decile_mean)

    # Convert the list of means into a tensor
    decile_means = torch.tensor(deciles)

    return decile_means
