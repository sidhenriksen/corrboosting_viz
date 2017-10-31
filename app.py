import dash,pickle
import dash_core_components as dcc
import dash_html_components as html
from scipy.stats import linregress
import plotly.graph_objs as go
import numpy as np
from plotly import tools

with open('data.pickle','rb') as f:
    data = pickle.load(f)

app = dash.Dash()
server = app.server
cellNames = list(data['cell'].keys())

def strip_uc(v):
    """
    This strips the duplicate uncorrelated values from the array and returns
    a vector with response values for correlated, anticorrelated, and uncorrelated.    
    
    """
    z = np.zeros(v.shape[1]*2+1)
    z[:-1] = v[[0,2],:].ravel()
    z[-1]= v[1,0]
    return z

    
def relative_ac_response(subData):
    '''
    Computes the relative anticorrelated response of a model/cell.

    Parameters
    ----------
    subData : dict of cell/model data (found in data.pickle)

    Returns
    --------
    slope : double, the relative anticorrelated response.

    '''
    
    x = subData['spikeCount'][2,:]
    y = subData['spikeCount'][0,:]
    slope,_,_,_,_ = linregress(x,y)
    return slope


def model_tuning_strength(cellData,modelData,correlated=1):
    '''
    Computes the model tuning strength (between cell and its GBEM unit).

    Parameters
    ----------
    cellData : dict of cell data (found in data.pickle)
    modelData: dict of model data (found in data.pickle)
    correlated : bool, optional. True if correlated, False if anticorrelated.

    Returns
    --------
    slope : double, the model tuning strength for given parameters

    '''
    
    idx = np.int64(correlated*2) # just an index
    
    x = cellData['spikeCount'][idx,:]
    
    y = modelData['spikeCount'][idx,:]
    
    slope,_,_,_,_ = linregress(x,y)
    
    return slope


def generate_tc_figure(cellData,lineStyle='line'):
    
    tcFig = [
        go.Scatter(
            x=cellData['dx'].ravel(),
            y=cellData['spikeCount'][0,:]/0.03,
            mode='lines',
            line=dict(width=4,color='rgb(20,20,20)',dash=lineStyle),
            name='Anticorrelated'),

        go.Scatter(
            x=cellData['dx'].ravel(),
            y=cellData['spikeCount'][2,:]/0.03,
            mode='lines',
            line = dict(width=4,color='rgb(200,20,20',dash=lineStyle),
            name='Correlated')
        ]
        
    return tcFig

def get_metrics(data,metric):

    
    if 'Cell' in metric: #capital C only for cell ac slope
        currentData = data['cell']
    elif 'Model' in metric:
        currentData = data['model']
    else:
        currentData = data
        
    names = data['cell'].keys()

    x = None
    
    # tc slope
    if 'tuning strength' in metric:
        isCorrelated = 'Correlated' in metric

        f = lambda name: model_tuning_strength(data['cell'][name],
                                                   data['model'][name],
                                                   correlated=isCorrelated)
        
        x = list(map(f,names))
        
        return x
    
    elif 'relative anticorrelated' in metric:

        f = lambda name:relative_ac_response(currentData[name])
        x = list(map(f,names))
        return x
    
    #warning('metric does not match any known metrics.')
    return x

def get_xlim(data,name):
    if 'dx' not in data['cell'][name]:
        import pdb; pdb.set_trace()
    xs = data['cell'][name]['dx']
    xlim = get_lims(xs)

def get_ylim(data,name):
    yMinCell = np.min(data['cell'][name]['spikeCount'])
    yMinModel = np.min(data['model'][name]['spikeCount'])
    
    yMaxCell = np.max(data['cell'][name]['spikeCount'])
    yMaxModel = np.max(data['model'][name]['spikeCount'])
    
    ylim = [np.min([yMinCell,yMinModel])/0.03,
            np.max([yMaxCell,yMaxModel])/0.03]
        
    return ylim
    
def get_lims(x,k=0.05):
    dx = np.max(x)-np.min(x)
    v = np.min(x) - k*.05
    w = np.max(x) + k*.05
    return [v,w]

currentCell = 'lemM328c12'

defaultDropdown = {'x-axis':'Correlated model tuning strength',
                    'y-axis':'Anticorrelated model tuning strength'}

app.layout = html.Div( children=[
    html.Div([html.Div(
            dcc.Dropdown(
                id=axisId,
                value=defaultDropdown[axisId],
                options=[{'label':'Correlated model tuning strength',
                              'value':'Correlated model tuning strength'},
                         {'label':'Anticorrelated model tuning strength',
                              'value':'Anticorrelated model tuning strength'},
                         {'label':'Cell relative anticorrelated response',
                              'value':'Cell relative anticorrelated response'},
                         {'label':'Model relative anticorrelated response',
                              'value':'Model relative anticorrelated response'}
                             ]),style={'width':'48%','display':'inline-block'}) \
                  for axisId in ['x-axis','y-axis']
    ],style={'width':'96%','display':'inline-block'}),
    html.Div([
        html.Div([
        dcc.Graph(id='main-graph',animate=False)],
        style={'width':'98%'}),
        html.Div([html.Center(html.Div([
            dcc.Graph(id='tc-graph')],style={'width':'99%'}))
                ],style={ 'width' : '100%'})
        ],style={'width':'100%','display':'inline-block'})
    ]
)


@app.callback(
    dash.dependencies.Output('main-graph','figure'),
    [dash.dependencies.Input('x-axis','value'),
     dash.dependencies.Input('y-axis','value')]
    )
def update_graph(xValue,yValue):

    if xValue is None:
        xValue = 'Correlated model tuning strength'

    if yValue is None:
        yValue = 'Anticorrelated model tuning strength'

        
    xData = np.round(get_metrics(data,xValue),3)
    yData = np.round(get_metrics(data,yValue),3)

    figure = {
        'data': [
            go.Scatter(
                x=xData,
                y=yData,
                mode='markers',
                text=cellNames,
                marker=dict(
                    line=dict(
                        width=2,
                        color='rgb(0,0,0)'
                        ),
                    size=20,
                    color='rgb(180,0,0)'
                    )
                )],
        'layout' : go.Layout(
            xaxis={'title':'\n'+xValue},
            yaxis={'title':yValue+'\n'},
            hovermode='closest',
            margin={'l':60,'t':30,'r':30,'b':60},
            font={'size':18}
            
        )
    }

    return figure



@app.callback(
    dash.dependencies.Output('tc-graph','figure'),
    [dash.dependencies.Input('main-graph','clickData')]
)
def update_tcs(clickData):
    if clickData is not None:
        currentCell=cellNames[clickData['points'][0]['pointNumber']]
    else:
        currentCell = 'lemM328c12'

    
    xlim = get_xlim(data,currentCell)
    ylim = get_ylim(data,currentCell)

    fig = tools.make_subplots(rows=1,cols=2)

    cellSubplotData = generate_tc_figure(data['cell'][currentCell])
    modelSubplotData = generate_tc_figure(data['model'][currentCell],lineStyle='dash')

    fig.append_trace(cellSubplotData[0],1,1)
    fig.append_trace(cellSubplotData[1],1,1)
    fig.append_trace(modelSubplotData[0],1,2)
    fig.append_trace(modelSubplotData[1],1,2)

    fig.layout.update({'xaxis1' : dict(zeroline=False,
                                         title='Disparity (deg)',
                                         range=xlim),
                       'xaxis2' : dict(zeroline=False,
                                         title='Disparity (deg)',
                                         range=xlim),
                       'yaxis1' : dict(zeroline=False,
                                          title='Spikes per second',
                                          range=ylim),
                       'yaxis2' : dict(zeroline=False,
                                           range=ylim),
                        'title':currentCell +
                        '                                GBEM' ,
                       'showlegend' : False,
                       'font':{'size':16},
                       'margin':{'t':50}
                     })

    
    return fig


if __name__ == "__main__":

    app.run_server(debug=True)


