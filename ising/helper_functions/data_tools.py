import copy

import numpy as np
import torch


# This code is currently unused in the IPR calculation
def extract_top_percent(tensor, percent):
    # Ensure the tensor is flattened
    tensor = tensor.view(-1)

    # Calculate the number of elements to keep (top percent)
    num_elements = int(percent * len(tensor))

    # Use torch.topk to get the top elements
    top_values, _ = torch.topk(tensor, num_elements, sorted=False, largest=True)

    return top_values


def calculate_ipr(model, r):
    with torch.no_grad():
        flat_parameters = [param.view(-1) for param in model.parameters()]
        flat_weights = torch.cat(flat_parameters)
        # flat_weights=extract_top_percent(flat_weights,weight_share)
        ipr_denom = torch.sqrt(torch.sum(flat_weights**2)) ** (2 * r)
        ipr_num = torch.sum(np.abs(flat_weights) ** (2 * r))
        ipr = ipr_num / ipr_denom
        return ipr.item()


def calculate_weight_norm(model, n):
    with torch.no_grad():
        flat_parameters = [param.view(-1) for param in model.parameters()]
        flat_weights = torch.cat(flat_parameters)
        weight_norm = torch.sum(flat_weights**n)
        return weight_norm


def calculate_cosine_similarity(model1, model2, weight_keys=None):
    cosine_tensor_list = []
    names_list = []
    with torch.no_grad():

        def prep_models(model):
            list_of_weights = []
            list_of_names = []
            for name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:
                    if not (param.shape[0] < 3 and len(param.shape) == 2):
                        list_of_weights.append(param)
                        list_of_names.append(name)

            flat_weights = torch.cat(
                [(torch.flatten(p)) for p in list_of_weights]
            )

            return list_of_weights, flat_weights, list_of_names

        list_of_weights_1, flattened_weights_1, weight_keys_1 = prep_models(
            model1
        )
        list_of_weights_2, flattened_weights_2, weight_keys_2 = prep_models(
            model2
        )

        flattened_cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(
            flattened_weights_1, flattened_weights_2
        )
        cosine_tensor_list.append(flattened_cosine.item())
        names_list.append("all model weights flattened")
        if weight_keys != None:
            for weight_key in weight_keys:
                index = weight_keys_1.index(weight_key)
                cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(
                    torch.flatten(list_of_weights_1[index]),
                    torch.flatten(list_of_weights_2[index]),
                )
                cosine_tensor_list.append(cosine.item())
                names_list.append(weight_key)
        cosine_tensor = torch.Tensor(cosine_tensor_list)

        return cosine_tensor, names_list


def evaluate_cosines(
    front_model, back_compare, epoch, weight_keys=None, compare_models=None
):
    # if epoch<=compare_models:
    #     return None
    if compare_models != None:
        back_compare.append(copy.deepcopy(front_model))
    if epoch > compare_models:
        cosines_tensor = calculate_cosine_similarity(
            back_compare[0], front_model, weight_keys=weight_keys
        )[0]

        # Assume `model_template` is an instance of the model's class (same architecture)
        back_steps_model = copy.deepcopy(
            back_compare[0]
        )  # Deep copy to preserve the architecture

        # Subtract the parameters
        with torch.no_grad():  # Disable gradient computation as we're manually updating parameters
            for p_new, p0, p1 in zip(
                back_steps_model.parameters(),
                back_compare[1].parameters(),
                back_compare[0].parameters(),
            ):
                p_new.copy_(p0 - p1)

        front_steps_model = copy.deepcopy(
            back_compare[-1]
        )  # Deep copy to preserve the architecture

        # Subtract the parameters
        with torch.no_grad():  # Disable gradient computation as we're manually updating parameters
            for p_new, p0, p1 in zip(
                front_steps_model.parameters(),
                front_model.parameters(),
                back_compare[-1].parameters(),
            ):
                p_new.copy_(p0 - p1)

        cosine_steps_tensor = calculate_cosine_similarity(
            back_steps_model,
            front_steps_model,
            weight_keys=weight_keys,
        )[0]

        # cosines.append(cosines_tensor)
        # cosine_steps.append(cosine_steps_tensor)

        # I think back_compare updated in-place.
        return (
            cosines_tensor,
            cosine_steps_tensor,
            back_steps_model,
            front_steps_model,
        )
    else:
        return None, None, None, None


# model1 back_model, model2 front_model


def linear_decomposition(model, criterion, data_loader, device="cpu"):
    with torch.no_grad():
        copied_model = copy.deepcopy(model)
        for batch in data_loader:
            loss_full = 0.0
            loss_linear = 0.0
            loss_non_linear = 0.0

            linear_norms = 0.0
            non_linear_norms = 0.0
            diff_norms = 0.0

            X_test, y_test = batch
            y_pred_full = copied_model(X_test).to(device)

            non_linear_y_pred = copied_model(X_test).to(device) + copied_model(
                -X_test
            ).to(device)
            linear_y_pred = y_pred_full - non_linear_y_pred

            float_y = y_test.float().clone().to(device)
            full_loss = criterion(y_pred_full, float_y)
            loss_full += full_loss.item()

            non_linear_loss = criterion(non_linear_y_pred, float_y)
            loss_non_linear += non_linear_loss.item()

            linear_loss = criterion(linear_y_pred, float_y)
            loss_linear += linear_loss.item()

            linear_norm = torch.sqrt(torch.sum(linear_y_pred**2)).item()
            linear_norms += linear_norm

            non_linear_norm = torch.sqrt(torch.sum(non_linear_y_pred**2)).item()
            non_linear_norms += non_linear_norm

            diff_norm = torch.sqrt(
                torch.sum((linear_y_pred - non_linear_y_pred) ** 2)
            ).item()
            diff_norms += diff_norm

    return (
        loss_full,
        loss_linear,
        loss_non_linear,
        linear_norms,
        non_linear_norms,
        diff_norms,
    )


def decile(back_steps_model, front_steps_model, split_into=10):
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
        decile_mean = ratio_differences[start_idx:end_idx].mean()  # .abs()?
        deciles.append(decile_mean)

    # Convert the list of means into a tensor
    decile_means = torch.tensor(deciles)

    return decile_means


def sample_indices(input_tensor, num_samples):
    # Get the total number of elements in the input tensor
    with torch.no_grad():
        dimensions = input_tensor.shape
        # Generate random indices for each dimension
        sampled_indices = torch.stack(
            [torch.randint(0, dim, (num_samples,)) for dim in dimensions],
            dim=-1,
        )
        # samples=input_tensor[tuple(sampled_indices.T)]#Clever - remember tuple is how you need to tuple to call the index from within the tensor so here you have a tensor of tuples.

    return sampled_indices


def linear_decomposition(model, criterion, dataloader, device):
    with torch.no_grad():
        copied_model = copy.deepcopy(model)
        test_loss_full = 0.0
        test_loss_linear = 0.0
        test_loss_non_linear = 0.0
        test_acc = 0.0
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


def evaluate_linear_decomposition(
    linear_decomposition_dict,
    network,
    loss_criterion,
    train_loader,
    test_loader,
    device,
):
    (
        full_loss_test,
        linear_loss_test,
        non_linear_loss_test,
        linear_norm_test,
        non_linear_norm_test,
        diff_norm_test,
    ) = linear_decomposition(network, loss_criterion, test_loader, device)
    linear_decomposition_dict["full_loss_test"].append(full_loss_test)
    linear_decomposition_dict["linear_loss_test"].append(linear_loss_test)
    linear_decomposition_dict["non_linear_loss_test"].append(
        non_linear_loss_test
    )
    linear_decomposition_dict["linear_norm_test"].append(linear_norm_test)
    linear_decomposition_dict["non_linear_norm_test"].append(
        non_linear_norm_test
    )
    linear_decomposition_dict["diff_norm_test"].append(diff_norm_test)

    (
        full_loss_train,
        linear_loss_train,
        non_linear_loss_train,
        linear_norm_train,
        non_linear_norm_train,
        diff_norm_train,
    ) = linear_decomposition(network, loss_criterion, train_loader, device)

    linear_decomposition_dict["full_loss_train"].append(full_loss_train)
    linear_decomposition_dict["linear_loss_train"].append(linear_loss_train)
    linear_decomposition_dict["non_linear_loss_train"].append(
        non_linear_loss_train
    )
    linear_decomposition_dict["linear_norm_train"].append(linear_norm_train)
    linear_decomposition_dict["non_linear_norm_train"].append(
        non_linear_norm_train
    )
    linear_decomposition_dict["diff_norm_train"].append(diff_norm_train)
