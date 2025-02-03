'''Code to import the NN run data, for Ising or ModAdd tasks, and compute information-geometric meassures.
...to run: Ensure filepaths for data are correct, ensure pwd is set to where the respective models are defined, run cells (#%% demarkation) sequentially to perform analysis and produce plots.
...please contact us to request data used in paper (>60GB).
'''
#Import libraries
import os
import numpy as np
import plotly.graph_objects as go
import torch

#%% # Import data
#Setup filepaths
baseroot = '.../Data/' #...filepath to data folder containing the NN run data
exp_root = 'ising' #...choose from: {'ising', 'modadd'}, filepath subfolder that specifies the task
#...Note set the pwd to be the folder where the NN class objects are defined (specific to the respective ising/modadd task)

#Setup import parameters
grokroots = ['Learning','Grokking']
runroots = [[run for run in os.listdir(baseroot+exp_root+'/'+grokroot) if run != '.DS_Store'] for grokroot in grokroots]
if exp_root == 'ising':
    filenameroot = 'save_blob.pt'
    epoch_max = 100000     #...final epoch NN run to
    epoch_plot_max = 50000 #...maximum epoch for plotting
    epoch_sample_stepsize = 200 
else:
    filenameroot = 'onerun.pt'
    epoch_max = 10000    #...final epoch NN run to
    epoch_plot_max = 200 #...maximum epoch for plotting
    epoch_sample_stepsize = 1

#Import the data
params, fims, train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], [], [], []
for g_idx, grokroot in enumerate(grokroots):
    params.append([])
    fims.append([])
    train_losses.append([])
    test_losses.append([])
    train_accuracies.append([])
    test_accuracies.append([])
    for runroot in runroots[g_idx]:
        file_path = baseroot+exp_root+'/'+grokroot+'/'+runroot+'/'+filenameroot
        full_data = torch.load(file_path) #...search data with full_data.__dict__.keys()
        epoch_paths = np.array(list(full_data.models.keys()))
        train_losses[-1].append(np.array(full_data.train_losses))
        test_losses[-1].append(np.array(full_data.test_losses))
        train_accuracies[-1].append(np.array(full_data.train_accuracies))
        test_accuracies[-1].append(np.array(full_data.test_accuracies))
        params[-1].append(np.array([np.concatenate([full_data.models[epoch]['model'][layer].flatten() for layer in full_data.models[epoch]['model'].keys()]) for epoch in epoch_paths]))
        fims[-1].append([np.array(torch.load(baseroot+exp_root+'/'+grokroot+'/'+runroot+f'/fims/epoch-{epoch}.pt')) for epoch in epoch_paths])

# Convert to arrays
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)
params = np.array(params)
fims = np.array(fims)
del(g_idx,grokroot,runroot,file_path,full_data)

###############################################################################
#%% #Compute the grok and learn times
def get_times(train_accs,test_accs,threshold_up=0.9,threshold_down=0.01):
    maxaccs=np.max(train_accs)
    above_acc=np.where(train_accs>threshold_up*maxaccs)[0]
    first_index_train=above_acc[0]

    maxaccs=np.max(test_accs)
    above_acc=np.where(test_accs>threshold_up*maxaccs)[0]
    first_index_test=above_acc[0]
    
    above_grok=np.where(test_accs>threshold_down*maxaccs)[0]
    first_index_above_grok=above_grok[0]
    
    return first_index_train,first_index_test,first_index_above_grok

#Compute the relevant boundary times for the periods
grok_train_time, grok_test_time, grok_above_grok_time = get_times(np.mean(train_accuracies,axis=1)[1],np.mean(test_accuracies,axis=1)[1])
learn_train_time, learn_test_time, learn_above_grok_time = get_times(np.mean(train_accuracies,axis=1)[0],np.mean(test_accuracies,axis=1)[0])

#%% #Define the plotting function
def create_pruning_plot(
    data: dict,
    title: str = None,
    axis_labels: tuple = ("X-Axis", "Y-Axis"),
    text_sizes: dict | None = None,
    save_path: str | None = None,
    template: str | None = "plotly",
    ylog: bool = False,
    xaxis_lims: tuple = None,
    yaxis_lims: tuple = None,
    legend_bool: bool = False
) -> go.Figure:
        
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
                showlegend=legend_bool
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
        template=template,
    )
    
    if ylog:
        fig.update_layout(
            yaxis_type="log"
            )
    if xaxis_lims is not None:
        fig.update_layout(
            xaxis=dict(range=[xaxis_lims[0], xaxis_lims[1]])
            )
    if yaxis_lims is not None:
        fig.update_layout(
            yaxis=dict(range=[yaxis_lims[0], yaxis_lims[1]])
            )
    if legend_bool: 
        fig.update_layout(
            legend=dict(
                font=dict(size=text_sizes.get("legend", 20)),
                title=dict(
                # text="Legend",
                font=dict(size=text_sizes.get("legend", 20))
                ),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
            )
        )
    else:
        fig.update_layout(showlegend=False)
            
    #if grok:
    fig.add_vline(x=grok_train_time, line=dict(color="black", width=1, dash="dash"))
    fig.add_vline(x=grok_test_time, line=dict(color="black", width=1, dash="dash"))
    fig.add_vline(x=grok_above_grok_time, line=dict(color="black", width=1, dash="dash"))
    #else:
    #fig.add_vline(x=learn_test_time, line=dict(color="black", width=1, dash="dash"))  

    # Save the plot as a PDF if a save_path is provided
    if save_path:
        fig.write_image(save_path, format="pdf", engine="kaleido")
        print(f"Plot saved as PDF: {save_path}")

    return fig

#%% #%% #Define cosine similarity functions
def cosine_similarity(vec1, vec2):
    # Compute the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the L2 norms (magnitudes) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    #print(dot_product,norm_vec1,norm_vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def fim_cosine_similarity(vec1, vec2, fim):
    # If only the diagonal is given convert to a matrix
    if len(fim.shape) == 1:
        fim = np.diag(fim)
        
    # Compute the dot product between the two vectors
    dot_product = np.matmul(vec1, np.matmul(fim, vec2))
    
    # Compute the L2 norms (magnitudes) of each vector
    norm_vec1 = np.sqrt(np.matmul(vec1, np.matmul(fim, vec1)))
    norm_vec2 = np.sqrt(np.matmul(vec2, np.matmul(fim, vec2)))
    #print(dot_product,norm_vec1,norm_vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

###############################################################################
#%% #Plot losses and accuracies
# Choose what to plot and whether to save
acc_idx = False #...False for losses, True for accuracies
file_name = 'AvgAccuracy' if acc_idx else 'AvgLosses'

fig = go.Figure()
if not acc_idx:
    data = {
        f'Train {grokroots[0]}': {"x": list(range(1,train_losses.shape[-1]+1)), "y": np.mean(train_losses[0, :, :], axis=0), "color": "blue"},
        f'Train {grokroots[1]}': {"x": list(range(1,train_losses.shape[-1]+1)), "y": np.mean(train_losses[1, :, :], axis=0), "color": "red"},
        f'Test {grokroots[0]}': {"x": list(range(1,test_losses.shape[-1]+1)), "y": np.mean(test_losses[0, :, :], axis=0), "color": "skyblue"},
        f'Test {grokroots[1]}': {"x": list(range(1,test_losses.shape[-1]+1)), "y": np.mean(test_losses[1, :, :], axis=0), "color": "orange"},
        }
else:
    data = {
        f'Train {grokroots[0]}': {"x": list(range(1,train_accuracies.shape[-1]+1)), "y": np.mean(train_accuracies[0, :, :], axis=0), "color": "blue"},
        f'Train {grokroots[1]}': {"x": list(range(1,train_accuracies.shape[-1]+1)), "y": np.mean(train_accuracies[1, :, :], axis=0), "color": "red"},
        f'Test {grokroots[0]}': {"x": list(range(1,test_accuracies.shape[-1]+1)), "y": np.mean(test_accuracies[0, :, :], axis=0), "color": "skyblue"},
        f'Test {grokroots[1]}': {"x": list(range(1,test_accuracies.shape[-1]+1)), "y": np.mean(test_accuracies[1, :, :], axis=0), "color": "orange"},
        }
    
fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", "Accuracy" if acc_idx else "Loss",),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog= not acc_idx,
     xaxis_lims = None,
     yaxis_lims = None #[-1,1]
)
fig.show()

###############################################################################
#%% #Compute average similarity scores of step updates
avg_fim_magnitudes, avg_fim_similarities = [], []
avg_euc_magnitudes, avg_euc_similarities = [], []
#n = 10 #...step-size
for g_idx in range(len(grokroots)):
    avg_fim_magnitudes.append([])
    avg_fim_similarities.append([])
    avg_euc_magnitudes.append([])
    avg_euc_similarities.append([])
    for r_idx in range(len(runroots[g_idx])):
        steps = params[g_idx,r_idx,1:,:]-params[g_idx,r_idx,:-1,:]
        avg_fim_magnitudes[-1].append(np.sqrt(np.einsum('ij,ij->i',steps,fims[g_idx,r_idx,:-1,:]*steps)))
        avg_fim_similarities[-1].append(np.einsum('ij,ij->i',steps[:-1,:],fims[g_idx,r_idx,1:-1,:]*steps[1:,:])/np.sqrt(np.einsum('ij,ij->i',steps[:-1,:],fims[g_idx,r_idx,1:-1,:]*steps[:-1,:])*np.einsum('ij,ij->i',steps[1:,:],fims[g_idx,r_idx,1:-1,:]*steps[1:,:])))
        avg_euc_magnitudes[-1].append(np.linalg.norm(steps,axis=-1))
        avg_euc_similarities[-1].append(np.einsum('ij,ij->i',steps[:-1],steps[1:])/(np.linalg.norm(steps[:-1],axis=-1)*np.linalg.norm(steps[1:],axis=-1)))
        '''#...undeveloped code for >1 samples per step (experimetation showed plot properties unchanged)
        #Step-size n>1 
        steps = params[g_idx,r_idx,n::n,:]-params[g_idx,r_idx,:-n:n,:]
        avg_fim_similarities[-1].append(np.einsum('ij,ij->i',steps[:-1,:],fims[g_idx,r_idx,n:-n:n,:]*steps[1:,:])/np.sqrt(np.einsum('ij,ij->i',steps[:-1,:],fims[g_idx,r_idx,n:-n:n,:]*steps[:-1,:])*np.einsum('ij,ij->i',steps[1:,:],fims[g_idx,r_idx,n:-n:n,:]*steps[1:,:])))
        '''
del(steps)

#Make arrays and average
avg_fim_magnitudes = np.mean(avg_fim_magnitudes,axis=1)
avg_fim_similarities = np.mean(avg_fim_similarities,axis=1)
avg_euc_magnitudes = np.mean(avg_euc_magnitudes,axis=1)
avg_euc_similarities = np.mean(avg_euc_similarities,axis=1)

#%% #Step FIM-Magnitude
fim_plot = True #...select whether to use the fisher measure (false is euclidean measure)
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'AvgMagnitues_FIM' if fim_plot else 'AvgMagnitues_Euc'
if not e_limit: file_name += '_Full'

fig = go.Figure()
if fim_plot:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_fim_magnitudes[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_fim_magnitudes[1], "color": "red"},
       }
else:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_euc_magnitudes[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_euc_magnitudes[1], "color": "red"},
       }
    
fig = create_pruning_plot(
     data,
     #axis_labels=("Epoch", "FIM Step Magnitude" if fim_plot else "Step Magnitude"),
     axis_labels=("Epoch", r'$|s|_{FIM}$' if fim_plot else r'$|s|$'),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=True,
     xaxis_lims = xaxis_lims,
     yaxis_lims = None, #[-1,1],
     legend_bool=True
)
fig.show()

#%% #Step Mag gradients
fim_plot = True #...select whether to use the fisher measure (false is euclidean measure)
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'AvgMagGrads_FIM' if fim_plot else 'AvgMagGrads_Euc'
if not e_limit: file_name += '_Full'
avg_fim_maggrads = (avg_fim_magnitudes[:,1:]-avg_fim_magnitudes[:,:-1])/epoch_sample_stepsize
avg_euc_maggrads = (avg_euc_magnitudes[:,1:]-avg_euc_magnitudes[:,:-1])/epoch_sample_stepsize

fig = go.Figure()
if fim_plot:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_fim_maggrads[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_fim_maggrads[1], "color": "red"},
       }
else:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_euc_maggrads[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_euc_maggrads[1], "color": "red"},
       }
    
fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", r'$d|s|_{FIM}/dt$' if fim_plot else "Step Magnitude Gradients"),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=False,
     xaxis_lims = xaxis_lims,
     yaxis_lims = None, #...other choices for ising/modadd: [-0.005, 0.005] or [-0.5,0.5]
     legend_bool=True
)
fig.show()

#%% #Step FIM-Cosine Similarity
fim_plot = True #...select whether to use the fisher measure (false is euclidean measure)
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'AvgSimilarities_FIM' if fim_plot else 'AvgSimilarities_Euc'
if not e_limit: file_name += '_Full'

fig = go.Figure()
if fim_plot:
    data = {
        grokroots[0]: {"x": epoch_paths[1:-1], "y": avg_fim_similarities[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[1:-1], "y": avg_fim_similarities[1], "color": "red"},
        #...use below when doing n>1 epochs between steps
        #grokroots[0]: {"x": epoch_paths[n:-n:n], "y": avg_fim_similarities[0], "color": "blue"},
        #grokroots[1]: {"x": epoch_paths[n:-n:n], "y": avg_fim_similarities[1], "color": "red"},
       }
else:
    data = {
        grokroots[0]: {"x": epoch_paths[1:-1], "y": avg_euc_similarities[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[1:-1], "y": avg_euc_similarities[1], "color": "red"},
       }
    
fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", r'$S_{C-FIM}(s^e,s^{e+1})$' if fim_plot else "Step Cosine Similarity"),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=False,
     xaxis_lims = xaxis_lims,
     yaxis_lims = [-1, 1],
     legend_bool=True
)
fig.show()

#%% #Position weight norms
fim_plot = False #...select whether to use the fisher measure (false is euclidean measure)
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'WeightNorms_FIM' if fim_plot else 'WeightNorms_Euc'
if not e_limit: file_name += '_Full'
avg_fim_wns = None ###
avg_euc_wns = np.mean(np.linalg.norm(params, axis=-1), axis=1)

fig = go.Figure()
if fim_plot:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_fim_wns[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_fim_wns[1], "color": "red"},
       }
else:
    data = {
        grokroots[0]: {"x": epoch_paths[:-1], "y": avg_euc_wns[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[:-1], "y": avg_euc_wns[1], "color": "red"},
       }
    
fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", "FIM Weight Norms" if fim_plot else "Weight Norms"),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=False,
     xaxis_lims = xaxis_lims,
     yaxis_lims = None #[-1, 1]
)
fig.show()

###############################################################################    
#%% #Cosine similarity vs the trajectory direct to the origin
avg_fim_OTsimilarities = []
avg_euc_OTsimilarities = []
for g_idx in range(len(grokroots)):
    avg_fim_OTsimilarities.append([])
    avg_euc_OTsimilarities.append([])
    for r_idx in range(len(runroots[g_idx])):
        steps = params[g_idx,r_idx,1:,:]-params[g_idx,r_idx,:-1,:]
        steps_towards_origin = -params[g_idx,r_idx,:-1,:]
        #Full versions using the functions
        #avg_fim_OTsimilarities[-1].append([fim_cosine_similarity(steps[i],steps_towards_origin[i],fims[g_idx,r_idx,i,:]) for i in range(len(steps))])
        #avg_euc_OTsimilarities[-1].append([cosine_similarity(steps[i],steps_towards_origin[i]) for i in range(len(steps))])
        #Hardcoded based on diagonal FIM assumption
        avg_fim_OTsimilarities[-1].append(np.einsum('ij,ij->i',steps,fims[g_idx,r_idx,:-1,:]*steps_towards_origin)/np.sqrt(np.einsum('ij,ij->i',steps,fims[g_idx,r_idx,:-1,:]*steps)*np.einsum('ij,ij->i',steps_towards_origin,fims[g_idx,r_idx,:-1,:]*steps_towards_origin)))
        avg_euc_OTsimilarities[-1].append(np.einsum('ij,ij->i',steps,steps_towards_origin)/(np.linalg.norm(steps,axis=-1)*np.linalg.norm(steps_towards_origin,axis=-1)))
del(steps,steps_towards_origin)

#Make arrays and average
avg_fim_OTsimilarities = np.mean(avg_fim_OTsimilarities,axis=1)
avg_euc_OTsimilarities = np.mean(avg_euc_OTsimilarities,axis=1)

#%% #Plotting
fim_plot = True #...select whether to use the fisher measure (false is euclidean measure)
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'AvgOTSimilarities_FIM' if fim_plot else 'AvgOTSimilarities_Euc'
if not e_limit: file_name += '_Full'

fig = go.Figure()
if fim_plot:
    data = {
        grokroots[0]: {"x": epoch_paths[1:-1], "y": avg_fim_OTsimilarities[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[1:-1], "y": avg_fim_OTsimilarities[1], "color": "red"},
       }
else:
    data = {
        grokroots[0]: {"x": epoch_paths[1:-1], "y": avg_euc_OTsimilarities[0], "color": "blue"},
        grokroots[1]: {"x": epoch_paths[1:-1], "y": avg_euc_OTsimilarities[1], "color": "red"},
       }
    
fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", r'$S_{C-FIM}(s^e,s^{OT})$' if fim_plot else "Step Cosine Similarity vs Origin Trajectory"),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=False,
     xaxis_lims = xaxis_lims,
     yaxis_lims = [-1, 1],
     legend_bool=False
)
fig.show()

###############################################################################    
#%% #Sloppy x Stiff splitting
# Plot the verage FIM-diag spectrum (to identify the cutoff ~0.01)
fig = go.Figure()
''' #...plot each seed's spectrum
for seed_idx in range(10):
    fig.add_trace(go.Scatter(y=np.sort(fims[0,seed_idx,-1,:]), mode='lines', name=f'{grokroots[0]}_{seed_idx}'))
    fig.add_trace(go.Scatter(y=np.sort(fims[1,seed_idx,-1,:]), mode='lines', name=f'{grokroots[1]}_{seed_idx}'))
'''
#...plot the seed-average spectrum
fig.add_trace(go.Scatter(y=np.mean(np.sort(fims[0,:,-1,:],axis=-1),axis=0), mode='lines', name=f'{grokroots[0]}'))
fig.add_trace(go.Scatter(y=np.mean(np.sort(fims[1,:,-1,:],axis=-1),axis=0), mode='lines', name=f'{grokroots[1]}'))

# Add a vertical line at epoch 4000
fig.add_hline(y=1e-1, line_dash="dash", line_color="black")

# Customize the layout
fig.update_layout(
    #title="Step `Magnitudes",
    xaxis_title="Parameter",
    yaxis_title="FIM-diagonal Values",
    yaxis_type="log",
    #xaxis=dict(range=[0, 0]),
    #yaxis=dict(range=[-3, 3]),
    showlegend=True,
    legend=dict(x=1.01, y=0.99)
)

# Save as a PDF file if needed
file_name = 'FIMdiags.pdf'
fig.write_image(baseroot+file_name)

#%% #Perform the sloppy x stiff splitting
#Set the threshold
threshold = 1e-1 #...FIM-diagonal values below this value have their respective parameters assigned to be sloppy

#Loop over the runs (to later average)
sloppy_step_magnitudes, sloppy_similarities = [], []
stiff_step_magnitudes, stiff_similarities = [], []
for g_idx in range(len(grokroots)):
    sloppy_step_magnitudes.append([])
    sloppy_similarities.append([])
    stiff_step_magnitudes.append([])
    stiff_similarities.append([])
    for r_idx in range(len(runroots[g_idx])):
        #Perform the split
        sloppy_idxs = np.where(fims[g_idx,r_idx,-1,:] < threshold)[0]
        stiff_idxs = np.where(fims[g_idx,r_idx,-1,:] >= threshold)[0]
        sloppy_params = params[g_idx,r_idx,:,sloppy_idxs].transpose()
        sloppy_fim = fims[g_idx,r_idx,:,sloppy_idxs].transpose()
        stiff_params = params[g_idx,r_idx,:,stiff_idxs].transpose()
        stiff_fim = fims[g_idx,r_idx,:,stiff_idxs].transpose()

        #FIM step analysis 
        #Sloppy submanifold
        sloppy_steps = sloppy_params[1:,:]-sloppy_params[:-1,:]
        #Full versions using the functions
        #sloppy_step_magnitudes[-1].append(np.array([np.sqrt(np.matmul(sloppy_steps[:,i], np.matmul(np.diag(sloppy_fim[:,i]), sloppy_steps[:,i]))) for i in range(sloppy_steps.shape[1])]))
        #sloppy_similarities[-1].append(np.array([fim_cosine_similarity(sloppy_steps[:,i],sloppy_steps[:,i+1],sloppy_fim[:,i+1]) for i in range(sloppy_steps.shape[1]-1)]))
        #Hardcoded based on diagonal FIM assumption
        sloppy_step_magnitudes[-1].append(np.sqrt(np.einsum('ij,ij->i',sloppy_steps,sloppy_fim[:-1,:]*sloppy_steps)))
        sloppy_similarities[-1].append(np.einsum('ij,ij->i',sloppy_steps[:-1,:],sloppy_fim[1:-1,:]*sloppy_steps[1:,:])/np.sqrt(np.einsum('ij,ij->i',sloppy_steps[:-1,:],sloppy_fim[1:-1,:]*sloppy_steps[:-1,:])*np.einsum('ij,ij->i',sloppy_steps[1:,:],sloppy_fim[1:-1,:]*sloppy_steps[1:,:])))
        
        #Stiff submanifold
        stiff_steps = stiff_params[1:,:]-stiff_params[:-1,:]
        #Full versions using the functions
        #stiff_step_magnitudes[-1].append(np.array([np.sqrt(np.matmul(stiff_steps[:,i], np.matmul(np.diag(stiff_fim[:,i]), stiff_steps[:,i]))) for i in range(stiff_steps.shape[1])]))
        #stiff_similarities[-1].append(np.array([fim_cosine_similarity(stiff_steps[:,i],stiff_steps[:,i+1],stiff_fim[:,i+1]) for i in range(stiff_steps.shape[1]-1)]))
        #Hardcoded based on diagonal FIM assumption
        stiff_step_magnitudes[-1].append(np.sqrt(np.einsum('ij,ij->i',stiff_steps,stiff_fim[:-1,:]*stiff_steps)))
        stiff_similarities[-1].append(np.einsum('ij,ij->i',stiff_steps[:-1,:],stiff_fim[1:-1,:]*stiff_steps[1:,:])/np.sqrt(np.einsum('ij,ij->i',stiff_steps[:-1,:],stiff_fim[1:-1,:]*stiff_steps[:-1,:])*np.einsum('ij,ij->i',stiff_steps[1:,:],stiff_fim[1:-1,:]*stiff_steps[1:,:])))
del(sloppy_idxs,stiff_idxs,sloppy_params,stiff_params,sloppy_fim,stiff_fim,sloppy_steps,stiff_steps)

#Take run averages
sloppy_step_magnitudes = np.mean(sloppy_step_magnitudes,axis=1)
sloppy_similarities    = np.mean(sloppy_similarities,axis=1)
stiff_step_magnitudes  = np.mean(stiff_step_magnitudes,axis=1)
stiff_similarities     = np.mean(stiff_similarities,axis=1)

#%% #Split Step FIM-Magnitude
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'SplitAvgMagnitues_FIM'
if not e_limit: file_name += '_Full'

fig = go.Figure()
data = {
    f'{grokroots[0]} Sloppy': {"x": epoch_paths[:-1], "y": sloppy_step_magnitudes[0], "color": "skyblue"},
    f'{grokroots[0]} Stiff': {"x": epoch_paths[:-1], "y": stiff_step_magnitudes[0], "color": "blue"},
    f'{grokroots[1]} Sloppy': {"x": epoch_paths[:-1], "y": sloppy_step_magnitudes[1], "color": "orange"},
    f'{grokroots[1]} Stiff': {"x": epoch_paths[:-1], "y": stiff_step_magnitudes[1], "color": "red"}   
   }

fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", r'$|s|_{FIM}$'),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=True,
     xaxis_lims = xaxis_lims,
     yaxis_lims = None,
     legend_bool=True
)
fig.show()

#%% #Split Step FIM-Cosine Similarity
e_limit = True #...set the x-axis epoch limit (false means no limit use all available)
if e_limit:
    xaxis_lims = [0, epoch_plot_max]
else:
    xaxis_lims = None
file_name = 'SplitAvgSimilarities_FIM'
if not e_limit: file_name += '_Full'

fig = go.Figure()
data = {
    f'{grokroots[0]} Sloppy': {"x": epoch_paths[1:-1], "y": sloppy_similarities[0], "color": "skyblue"},
    f'{grokroots[0]} Stiff': {"x": epoch_paths[1:-1], "y": stiff_similarities[0], "color": "blue"},
    f'{grokroots[1]} Sloppy': {"x": epoch_paths[1:-1], "y": sloppy_similarities[1], "color": "orange"},
    f'{grokroots[1]} Stiff': {"x": epoch_paths[1:-1], "y": stiff_similarities[1], "color": "red"}   
   }

fig = create_pruning_plot(
     data,
     #title="Pruning curves for weight decay 3e-05, bs 64, P 113",
     axis_labels=("Epoch", r'$S_{C-FIM}(s^e,s^{e+1})$'),
     text_sizes={"title": 24, "axis_labels": 18, "legend": 14},
     save_path=baseroot+file_name+".pdf",
     ylog=False,
     xaxis_lims = xaxis_lims,
     yaxis_lims = [-1, 1],
     legend_bool=True
)
fig.show()

