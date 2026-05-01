import plotly.colors as plc
import plotly.graph_objects as go


def pick_colours(wm, min_wm, max_wm) -> list:
    """
    Function to pick colours according to a function on the datadict.
    For example, you may want to color according to the grokking time.
    The input data_dict should be the dict that will be used for the plots
    i.e. the keys are the different lines (e.g. different weight multipliers)
    and the values are the data that will be plotted (e.g. the test/train accuracies)
    """
    combined_colorscale = [
        "rgb(0, 0, 255)",
        "rgb(204, 204, 255)",
        "rgb(255, 0, 0)",
    ]
    max_colour = max_wm
    min_colour = min_wm
    epsilon = 1e-6
    normed_color = (wm - min_colour) / (max_colour - min_colour + epsilon)
    if normed_color == 1:
        normed_color -= epsilon
    if normed_color == 0:
        normed_color += epsilon
    colour_key = plc.sample_colorscale(combined_colorscale, normed_color)[0]
    return colour_key


def value_to_rgb(value, lower, upper):
    """
    Convert a value within a specified range to a HEX color transitioning from blue to red.

    Args:
        value (int): A value within the range specified by lower and upper.
        lower (int): The lower bound of the range.
        upper (int): The upper bound of the range.

    Returns:
        str: A HEX string representing the color.
    """
    value = float(value)
    if not (lower <= value <= upper):
        raise ValueError(
            f"Value must be between {lower} and {upper}, inclusive."
        )

    # Normalize value to a range of 0 to 1
    normalized = (value - lower) / (upper - lower)

    # Interpolate between blue (0, 0, 255) and red (255, 0, 0)
    red = int(normalized * 255)
    green = 0
    blue = int((1 - normalized) * 255)

    # Convert to HEX format
    return f"#{red:02x}{green:02x}{blue:02x}"


def create_pruning_plot(
    data: dict,
    title: str = "Pruning Curves",
    axis_labels: tuple = ("X-Axis", "Y-Axis"),
    text_sizes: dict | None = None,
    save_path: str | None = None,
    template: str | None = "plotly",
) -> go.Figure:
    """
    Create a pruning curve plot using Plotly with customizable text sizes and
    optional saving as PDF.

    Args:
        data (dict): Dictionary where keys are legends and values are
            dictionaries containing 'x' (prune percentages) and 'y'
            (accuracies).
                     Example:
                     {
                         "0.1": {"x": [...], "y": [...]},
                         "0.2": {"x": [...], "y": [...]},
                     }
        title (str): Title of the plot.
        axis_labels (tuple): A tuple containing x-axis and y-axis labels
            (x_label, y_label).
        text_sizes (dict): Dictionary to specify text sizes for the plot elements.
                           Example: {
                               "title": 20,
                               "axis_labels": 15,
                               "legend": 12
                           }
        save_path (str): File path to save the plot as a PDF. If None, the plot
            is not saved.
        template (str): The plotly template to use, e.g. "plotly_white".
            Default "plotly".

    Returns:
        go.Figure: A Plotly figure object.
    """
    # Default text sizes if not provided
    if text_sizes is None:
        text_sizes = {"title": 20, "axis_labels": 20, "legend": 20}

    x_label, y_label = axis_labels

    fig = go.Figure()

    # Add traces for each legend in the data
    for legend, values in data.items():
        fig.add_trace(
            go.Scatter(
                x=values["x"],
                y=values["y"],
                mode="lines",
                name=legend,  # Legend label
                line=dict(color=values.get("color", None), dash="solid"),
            )
        )

    # Update layout and axes with custom text sizes
    fig.update_layout(
        title=dict(text=title, font=dict(size=text_sizes.get("title", 20))),
        xaxis=dict(
            title=dict(
                text=x_label, font=dict(size=text_sizes.get("axis_labels", 20))
            ),
            tickfont=dict(size=text_sizes.get("ticks", 20)),  # Add this line for x-axis tick labels
            # range=[0.5, 1],
        ),
        yaxis=dict(
            title=dict(
                text=y_label, font=dict(size=text_sizes.get("axis_labels", 20))
            ),
            # range=[0, 1],
            tickfont=dict(size=text_sizes.get("ticks", 20)),  # Add this line for x-axis tick labels
        ),
        legend=dict(
            font=dict(size=text_sizes.get("legend", 20)),
            title=dict(
                # text="Legend",
                font=dict(size=text_sizes.get("legend", 20))
            ),
        ),
        template=template,
    )

    # Save the plot as a PDF if a save_path is provided
    if save_path:
        fig.write_image(save_path, format="pdf", engine="kaleido")
        print(f"Plot saved as PDF: {save_path}")

    return fig


# # Example usage with hypothetical data
# example_data = {
#     "0.1": {"x": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "y": [1, 0.95, 0.9, 0.85, 0.8, 0.75], "color": "blue"},
#     "0.2": {"x": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "y": [1, 0.94, 0.88, 0.82, 0.78, 0.72], "color": "lightblue"},
#     "10.0": {"x": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "y": [1, 0.7, 0.5, 0.4, 0.3, 0.2], "color": "red"},
# }
#
# # Generate the plot and save it as a PDF
# fig = create_pruning_plot(
#     example_data,
#     title="Pruning curves for weight decay 3e-05, bs 64, P 113",
#     axis_labels=("Prune percents", "Accuracy"),
#     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
#     save_path="pruning_curves.pdf"
# )
# fig.show()
#
