import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point
import matplotlib as mpl


GEOGRAPHICAL_PROJECTION = {'Italy':'EPSG:7791','USA':'EPSG:2163'}

#-------------------------------------------------------------------------- 
# Plots the flow map for standard f
#-------------------------------------------------------------------------- 
def map_flows(gdf_flows,gdf_nodes,output_path,colormap='viridis_r',node_size='population',max_alpha_links = 1.0, min_alpha_links = 0.5,steps_alpha_links = 10, rescale_nodesize = 200,
             vmin = None, vmax = None,min_flow = 0,legend_shift = 0):
    
        
    #Internal parameters
    column = 'flow'
    node_A = 'from'
    node_B = 'to'
    zlabel = 'flows'
    
    if vmin == None:
        vmin = gdf_flows[column].min()
    if vmax == None:
        vmax = gdf_flows[column].max()

    #removes zeros and self loops
    gdf_flows = gdf_flows[gdf_flows[column] > min_flow]
    gdf_flows = gdf_flows[gdf_flows[node_A] != gdf_flows[node_B]]

    #sort to better visual and cuts in levels for alpha
    gdf_flows = gdf_flows.sort_values(by=column)
    gdf_flows['cuts'] = pd.qcut(gdf_flows[column],steps_alpha_links)

    #normalize
    norm_flow = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    norm_pop  = mpl.colors.PowerNorm(gamma=0.5,vmin=vmin,vmax=vmax)

    fig,ax = plt.subplots(figsize=(10,10))

    gdf_nodes.boundary.plot(ax=ax,lw=0.5,color=(.7,.7,.7))


    alpha0 = min_alpha_links
    for i,gdf_cut in gdf_flows.groupby('cuts'):
        gdf_cut.plot(ax=ax,column = column,cmap=colormap,norm=norm_flow,alpha=alpha0,lw=1)
        alpha0 += (max_alpha_links-min_alpha_links)/(steps_alpha_links-1)
        if alpha0 > max_alpha_links:
            alpha0 = max_alpha_links

    gdf_nodes.centroid.plot(ax=ax,color=(.7,.7,.7),markersize=rescale_nodesize*norm_pop(gdf_nodes[node_size]))

    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    #Colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.8+legend_shift, 0.4, 0.025, 0.3])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm_flow)
    sm._A = []
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel(zlabel,fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(output_path,bbox_inches = 'tight', transparent=False,dpi=300)
    
    
