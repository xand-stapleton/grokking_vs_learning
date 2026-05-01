import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TrainingPlot:
    def __init__(self, plot_enabled=False):
        self.plot_enabled = plot_enabled

        self.fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Accuracy", "Loss")
        )
        if self.plot_enabled:
            # Initialize traces for loss and accuracy
            self.test_acc_trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Test accuracy",
                line=dict(color="red", dash="solid"),
            )
            self.train_acc_trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Train accuracy",
                line=dict(color="blue", dash="dash"),
            )

            self.test_loss_trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Test loss",
                line=dict(color="red", dash="solid"),
            )
            self.train_loss_trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Train loss",
                line=dict(color="blue", dash="dash"),
            )

            # Add traces to the figure
            self.fig.add_trace(self.test_acc_trace, row=1, col=1)
            self.fig.add_trace(self.train_acc_trace, row=1, col=1)

            self.fig.add_trace(self.test_loss_trace, row=1, col=2)
            self.fig.add_trace(self.train_loss_trace, row=1, col=2)

            self.fig.update_xaxes(title_text="Epoch")
            self.fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            self.fig.update_yaxes(title_text="Loss", row=1, col=1)


def plot_traincurves(
    epochs,
    test_accuracies,
    train_accuracies,
    test_losses,
    train_losses,
    config_dict,
):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    test_acc_trace = go.Scatter(
        x=epochs,
        y=test_accuracies,
        mode="lines",
        name="Test accuracy",
        line=dict(color="red", dash="solid"),
    )
    train_acc_trace = go.Scatter(
        x=epochs,
        y=train_accuracies,
        mode="lines",
        name="Train accuracy",
        line=dict(color="blue", dash="dash"),
    )

    test_loss_trace = go.Scatter(
        x=epochs,
        y=test_losses,
        mode="lines",
        name="Test loss",
        line=dict(color="red", dash="solid"),
    )
    train_loss_trace = go.Scatter(
        x=epochs,
        y=train_losses,
        mode="lines",
        name="Train loss",
        line=dict(color="blue", dash="dash"),
    )

    # Add traces to the figure
    fig.add_trace(test_acc_trace, row=1, col=1)
    fig.add_trace(train_acc_trace, row=1, col=1)

    fig.add_trace(test_loss_trace, row=1, col=2)
    fig.add_trace(train_loss_trace, row=1, col=2)

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2, type="log")

    fig.update_layout(
        title_text=f'mlp {config_dict["hidden"]}, wd {config_dict["weight_decay"]}, lr {learning_rate}, multiplier {config_dict["weight_multiplier"]}, train frac {train_fraction}, optimizer {str(config_dict["optimizer"])}'
    )

    # ,  )

    return fig


# code for decile plot
def decile_plot(saved_object, nongrokked_object):
    # Define the number of lines
    split_into = save_object.decile_means.shape[-1]
    num_lines = split_into

    # Create a color scale from red to very pale blue
    colorscale = [
        "rgb(255, 0, 0)",  # Red
        "rgb(255, 153, 153)",  # Light red
        "rgb(204, 204, 255)",  # Very pale blue
    ]

    # Generate a set of colors for each line
    colors = pc.sample_colorscale(colorscale, num_lines)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[])
    for i in range(split_into):
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(saved_object.decile_means[:, i])),
                y=saved_object.decile_means[:, i],
                mode="lines",
                line=dict(color=colors[i], width=2),
                name=f"{100*(i/split_into):.0f}%-"
                + f"{100*((i+1)/split_into):.0f}%"
                + f" weights by magnitude",
            ),
            row=1,
            col=1,
        )
        fig.update_layout(
            title_text=f"Ratio of weight update to initial weight magnitude split by weight magnitude, step (m)=1"
        )
        return fig

    decile_plot(save_object, save_object).show()

    def weight_sample_plot(saved_object, nongrokked_object):
        # Define the number of lines

        weight_samples = saved_object.weight_samples.shape[-1]
        layers = saved_object.weight_samples.shape[1]

        num_lines = weight_samples

        # Create a color scale from red to very pale blue
        colorscale = [
            "rgb(255, 0, 0)",  # Red
            "rgb(255, 153, 153)",  # Light red
            "rgb(204, 204, 255)",  # Very pale blue
        ]

        # Generate a set of colors for each line
        colors = pc.sample_colorscale(colorscale, num_lines)

        fig = make_subplots(rows=2, cols=1, subplot_titles=[])
        # fig=make_subplots(rows=2,cols=layers,subplot_titles=['Whole network random weight sample']+[f'Layer {i+1}' for i in range(layers)],specs=[[{"colspan": layers}, None], [None] * layers])

        for i in range(weight_samples):
            # whole network:
            for j in range(layers):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(
                            0, len(saved_object.weight_samples[:, j, i])
                        ),
                        y=saved_object.weight_samples[:, j, i],
                        mode="lines",
                        line=dict(color=colors[i], width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                # fig.add_trace(go.Scatter(x=np.arange(0,len(saved_object.weight_samples[:,j,i])),y=saved_object.weight_samples[:,j,i],mode='lines',line=dict(color=colors[i], width=2),showlegend=False),row=2,col=1+j)
                # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(saved_object.test_accuracies))),
                        y=saved_object.test_accuracies,
                        mode="lines",
                        line=dict(color="blue", dash="solid"),
                        showlegend=True,
                        name=r"$\text{Test}$",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(saved_object.train_accuracies))),
                        y=saved_object.train_accuracies,
                        mode="lines",
                        line=dict(color="blue", dash="dash"),
                        showlegend=True,
                        name=r"$\text{Train}$",
                    ),
                    row=2,
                    col=1,
                )
                fig.update_yaxes(title_text="Accuracy", row=2, col=1)
                fig.update_yaxes(title_text="Sampled weight", row=1, col=1)
                fig.update_xaxes(title_text="Epochs", row=1, col=1)
                fig.update_layout(
                    title_text=f"Random weight samples, samples {saved_object.weight_samples.shape[-1]}, lr={saved_object.trainargs.lr}, wd={saved_object.trainargs.weight_decay}, wm={saved_object.trainargs.weight_multiplier}"
                )
                return fig
