
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
import numpy as np 
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

##############################################################################

def efficient_series_ofsequences(length_of_sequence, length_of_series):
    BASES = np.array(['A', 'C', 'T', 'G'])
    P = np.array([0.2, 0.3, 0.2, 0.3])
    # random DNA bases
    sequences_array = np.random.choice(BASES, (length_of_series, length_of_sequence), p=P)
    return sequences_array

def find_homopolymers(sequence):
    homopolymers = []
    for match in re.finditer(r"(A+|C+|T+|G+)", ''.join(sequence)):
        if len(match.group()) > 1:  # considering stretches longer 2, >=
            homopolymers.append((match.start(), match.end()))
    return homopolymers

def introduce_errors_optimised(sequence, error_rate):
    BASES = np.array(['A', 'C', 'T', 'G'])
    error_types = ['insertion', 'mismatch', 'deletion']
    
    error_positions = {etype: [] for etype in error_types}
    
    num_errors = int(len(sequence) * error_rate)
    
    # bias errors towards the end
    probs = np.linspace(0.5, 1.5, len(sequence))
    
    # Increase error probability on homopolymers
    homopolymers = find_homopolymers(sequence)
    for start, end in homopolymers:
        probs[start:end] *= 2  # double the error probability for homopolymer regions
    
    probs /= probs.sum()
    
    positions = np.random.choice(len(sequence), size=num_errors, p=probs, replace=False)
    chosen_errors = np.random.choice(error_types, size=num_errors)
    
    
    sorted_indices = np.argsort(positions)
    positions = positions[sorted_indices]
    chosen_errors = chosen_errors[sorted_indices]
    
    shift = 0  # to account for insertions and deletions
    
    for idx, pos in enumerate(positions):
        adjusted_pos = pos + shift
        error_type = chosen_errors[idx]
        
        if error_type == 'insertion':
            random_base = np.random.choice(BASES)
            sequence = np.insert(sequence, adjusted_pos, random_base)
            error_positions['insertion'].append(adjusted_pos)
            shift += 1  # account for the new insertion
            
        elif error_type == 'mismatch':
            original_base = sequence[adjusted_pos]
            replacement_bases = [base for base in BASES if base != original_base]
            sequence[adjusted_pos] = np.random.choice(replacement_bases)
            error_positions['mismatch'].append(adjusted_pos)
            
        elif error_type == 'deletion':
            sequence = np.delete(sequence, adjusted_pos)
            error_positions['deletion'].append(adjusted_pos)
            shift -= 1  # account for the deletion
    
    return sequence, error_positions


def run_code(error_rates, length_of_sequence, length_of_series):
    results = {}
    
    for rate in error_rates:
        corrupted_sequences_for_rate = []
        error_positions_for_rate = []
        sequence_set = efficient_series_ofsequences(length_of_sequence, length_of_series)
        
        for seq in sequence_set:
            corrupted_seq, error_positions = introduce_errors_optimised(seq, rate)
            corrupted_sequences_for_rate.append(corrupted_seq)
            error_positions_for_rate.append(error_positions)
        
        results[rate] = {
            'corrupted_sequences': corrupted_sequences_for_rate,
            'error_positions': error_positions_for_rate
        }
    
    return results


def find_homopolymer_positions(sequence):
    BASES = ['A', 'C', 'G', 'T']
    positions = {base: [] for base in BASES}
    for base in BASES:
        for match in re.finditer(f"{base}+", ''.join(sequence)):
            if len(match.group()) > 1:  # considering stretches 2 or longer
                positions[base].extend(range(match.start(), match.end()))
    return positions
##############################################################################
#plots : 
def generate_error_distribution_plot(error_type, error_rates, results):
    num_rates = len(error_rates)
    
    fig = make_subplots(rows=1, cols=num_rates, subplot_titles=[f'Error rate: {rate*100}%' for rate in error_rates])
    
    for i, rate in enumerate(error_rates):
        error_positions = results[rate]['error_positions']
        positions = [pos for eps in error_positions for pos in eps[error_type]]
        
        trace = go.Histogram(x=positions, name=f'Error rate: {rate*100}%', histnorm='probability')
        fig.add_trace(trace, row=1, col=i+1)
    
    
    fig.update_layout(title_text=f'Positions for {error_type.capitalize()} errors')
    fig.update_xaxes(title_text="Position in Sequence", range=[-5, 105])  # setting the x-axis range
    fig.update_yaxes(title_text="Probability", range=[0, 0.5])  # setting the y-axis range
    
    return fig


def generate_length_distribution_plot(error_rates, results):
    fig = make_subplots(rows=1, cols=len(error_rates))
    
    for i, rate in enumerate(error_rates):
        lengths = [len(seq) for seq in results[rate]['corrupted_sequences']]
        
        trace = go.Histogram(x=lengths, name=f'Error Rate: {rate*100}%')
        fig.add_trace(trace, row=1, col=i+1)
        fig.update_layout(title_text=f'Sequence length distribution per error rate')
        fig.update_xaxes(title_text='Sequence Length', range=[80, 120], row=1, col=i+1)  # setting the x-axis range
        fig.update_yaxes(title_text='Number of Sequences', range=[0, 45], row=1, col=i+1)  # setting the y-axis range
    
    return fig

def compute_gc_content(sequence):
    """Compute the GC content of a sequence."""
    sequence = np.array(list(sequence))  # converting string sequence to numpy array for efficient computation
    num_g_c = np.sum(sequence == 'G') + np.sum(sequence == 'C')
    return (num_g_c / len(sequence)) * 100

def analyse_gc_content(results):
    analysis = {}
    
    for rate in results:
        gc_contents = [compute_gc_content(seq) for seq in results[rate]['corrupted_sequences']]
        analysis[rate] = {
            'average_gc_content': np.mean(gc_contents),
            'std_dev_gc_content': np.std(gc_contents)
        }
    
    return analysis

def find_homopolymers(seq):
    homopolymers = []
    curr_base = None
    curr_start = None
    for i, base in enumerate(seq):
        if base == curr_base:
            continue
        if curr_base is not None:
            homopolymers.append((curr_start, i))
        curr_base = base
        curr_start = i
    if curr_base is not None:
        homopolymers.append((curr_start, len(seq)))
    return homopolymers

def calc_homopolymer_ratios(sequences, error_rates):
    ratios = {rate:{'A':[], 'C':[], 'T':[], 'G':[]} for rate in error_rates}
    for rate in error_rates:
        for seq in sequences[rate]['corrupted_sequences']:
            hp = find_homopolymers(seq)
            total_len = len(seq)
            for start, end in hp:
                length = end - start
                base = seq[start]
                ratios[rate][base].append(length/total_len)
    return ratios



def plotly_gc_content_against_error_rate(analysis):
    """Generate a Plotly figure showing average GC content against the error rate."""
    
    error_rates = list(analysis.keys())
    avg_gc_contents = [analysis[rate]['average_gc_content'] for rate in error_rates]
    std_devs = [analysis[rate]['std_dev_gc_content'] for rate in error_rates]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=error_rates, 
        y=avg_gc_contents, 
        mode='lines+markers',
        error_y=dict(
            type='data', 
            array=std_devs,
            visible=True, 
            thickness=1.5,
            width=3,
        ),
        line=dict(color='green'),
        name='Average GC Content'
    ))
    
    fig.update_layout(
        title='Average GC Content vs. Error Rate',
        xaxis_title='Error Rate',
        yaxis_title='Average GC Content (%)',
        plot_bgcolor='white'
    )

    return fig

def plotly_ratios(ratios, error_rates):
    """Generates a Plotly figure for Homopolymer Ratios by Error Rate."""
    
 
    fig = go.Figure()
    
    # looping through the bases and add them to the plot
    for base in 'ACGT':
        means = [np.mean(ratios[rate][base]) for rate in error_rates]
        fig.add_trace(go.Scatter(x=error_rates, y=means, mode='lines+markers', name=base))
    
    # updating the layout
    fig.update_layout(
        title='Homopolymer Ratios by Error Rate',
        xaxis_title='Error Rate',
        yaxis_title='Homopolymer Ratio',
        legend_title='Base',
        template="plotly_white"
    )
    
    return fig




def plotly_homopolymer_positions_heatmap(results):
    BASES = ['A', 'C', 'G', 'T']    
    error_rates = sorted(results.keys())
    sequence_length = len(results[error_rates[0]]['corrupted_sequences'][0])
    
    #subplot for each error rate
    fig = make_subplots(rows=len(error_rates), cols=1,
                        subplot_titles=[f'Error Rate: {rate*100}%' for rate in error_rates])
    
    for rate_idx, rate in enumerate(error_rates):
        #matrix to store homopolymer frequencies for each base
        homopolymer_frequencies = np.zeros((len(BASES), sequence_length))
        
        num_sequences = len(results[rate]['corrupted_sequences'])
        
        for seq in results[rate]['corrupted_sequences']:
            positions = find_homopolymer_positions(seq)
            for base_idx, base in enumerate(BASES):
                for pos in positions[base]:
                    if pos < sequence_length:  
                        homopolymer_frequencies[base_idx, pos] += 1
        
        # normalise frequencies by the number of sequences
        homopolymer_frequencies /= num_sequences
        
        # plotly plot
        heatmap = go.Heatmap(z=homopolymer_frequencies, x=list(range(sequence_length)), y=BASES,
                             colorscale='Hot', showscale=True)
        fig.add_trace(heatmap, row=rate_idx+1, col=1)
        
    fig.update_layout(height=300*len(error_rates), title_text="Homopolymer Positions by Error Rate")
    return fig



##############################################################################


error_rates = [0.02, 0.05, 0.1]
results = run_code(error_rates, 100, 100)

#showing the GC content per error rate
analysis = analyse_gc_content(results)
gc_content_figure = plotly_gc_content_against_error_rate(analysis)
ratios = calc_homopolymer_ratios(results, error_rates) 
ratios_figure = plotly_ratios(ratios, error_rates)
heatmap_figure = plotly_homopolymer_positions_heatmap(results)

##############################################################################
# Dash App initialisation
app = dash.Dash(__name__)
server = app.server


error_distribution_plots = {}
for error_type in ['insertion', 'deletion', 'mismatch']:
    error_distribution_plots[error_type] = generate_error_distribution_plot(error_type, error_rates, results)



# App layout
app.layout = html.Div([
    # description section
    html.Div([
        html.H1("DNA Error Rate Dashboard", className="dashboard-title"),
        html.P("This dashboard allows interactive exploration of how error types and rates impact generated sequence properties like length, GC content and homopolymer ratios", className="dashboard-description"),
    ], className="description-section", style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '50px'}),
    
        
    html.Div([
        html.Label("Error Rates (comma-separated):"),
        dcc.Input(id="input-error-rates", value="0.02,0.05,0.1", type="text"),
        
        html.Label("Number of Sequences:"),
        dcc.Input(id="input-num-sequences", value="100", type="number"),
        
        html.Label("Amount of Bases:"),
        dcc.Input(id="input-num-bases", value="100", type="number"),
        
        html.Button("Generate Data and Update Plots", id="submit-button"),
    ], style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}),        
        
    # Error Distribution plots
        
        
    html.Div([
        dcc.Dropdown(
            id='error-type-dropdown',
            options=[
                {'label': 'Insertion', 'value': 'insertion'},
                {'label': 'Deletion', 'value': 'deletion'},
                {'label': 'Mismatch', 'value': 'mismatch'}
            ],
            value='insertion',
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        dcc.Graph(id='error-distribution-graph', figure=error_distribution_plots['insertion']),
    ], style={'width': '50%', 'margin-top': '20px', 'margin-bottom': '50px', 'display': 'inline-block'}),
        
    
    html.Div([
        dcc.Graph(id='length-distribution-graph', figure=generate_length_distribution_plot(error_rates, results))
    ], style={'width': '50%', 'margin-top': '20px', 'margin-bottom': '50px', 'display': 'inline-block'}),

    html.Div([
        html.H3("GC Content vs. Error Rate", className="plot-title"),
        dcc.Graph(id='gc-content-plot', figure=gc_content_figure),
    ], className="gc-content-section", style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '50px'}),
    
    html.Div([
        html.H3("Homopolymer Ratios by Error Rate", className="plot-title"),
        dcc.Graph(id='ratios-plot', figure=ratios_figure),
    ], className="ratios-section", style={'width': '40%', 'margin-top': '20px', 'margin-bottom': '50px', 'display': 'inline-block'}, ),

    html.Div([
        html.H3("Homopolymer Positions by Error Rate", className="plot-title"),
        dcc.Graph(id='homopolymer-heatmap', figure=heatmap_figure),
    ], className="heatmap-section", style={'width': '60%', 'margin-top': '20px', 'margin-bottom': '50px', 'display': 'inline-block'}),


            
        
        
])
#####################################################################################################################################
@app.callback(
    Output('error-distribution-graph', 'figure'),
    [Input('error-type-dropdown', 'value'),
     Input('submit-button', 'n_clicks')],
    [State('input-error-rates', 'value'),
     State('input-num-sequences', 'value'),
     State('input-num-bases', 'value')]
)
def update_error_distribution_plot(error_type, n, error_rates_str, num_sequences, num_bases):
    error_rates = [float(er) for er in error_rates_str.split(",")]
    results = run_code(error_rates, int(num_sequences), int(num_bases))
    error_distribution_figure = generate_error_distribution_plot(error_type, error_rates, results)
    return error_distribution_figure


@app.callback(
    [
        Output('length-distribution-graph', 'figure'),
        Output('gc-content-plot', 'figure'),
        Output('ratios-plot', 'figure'),
        Output('homopolymer-heatmap', 'figure'),
    ],
    [Input('submit-button', 'n_clicks')],
    [State('input-error-rates', 'value'),
     State('input-num-sequences', 'value'),
     State('input-num-bases', 'value')]
)
def update_all_plots_except_error(n, error_rates_str, num_sequences, num_bases):
    error_rates = [float(er) for er in error_rates_str.split(",")]
    results = run_code(error_rates, int(num_sequences), int(num_bases))

    # updating the plots (excluding the error distribution plot)
    length_distribution_figure = generate_length_distribution_plot(error_rates, results)
    gc_content_figure = plotly_gc_content_against_error_rate(analyse_gc_content(results))
    ratios_figure = plotly_ratios(calc_homopolymer_ratios(results, error_rates), error_rates)
    heatmap_figure = plotly_homopolymer_positions_heatmap(results)
    
    return length_distribution_figure, gc_content_figure, ratios_figure, heatmap_figure



if __name__ == '__main__':
    app.run_server(debug=True)