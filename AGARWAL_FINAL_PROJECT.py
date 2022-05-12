# Import the required libraries
from numpy import NaN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib.ticker import StrMethodFormatter
# part 1 - Data Cleaning and Processing
#Create an empty dataframe
df_new = pd.DataFrame()
#create a list of states
state_list = ['Alaska','Alabama','Arkansas','Arizona','California','Colorado','Connecticut','District of Columbia','Delaware','Florida','Georgia','Hawaii','Iowa','Idaho','Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri','Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey','New Mexico','Nevada','New York','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming', 'Puerto Rico']

#Run a loop on all the excel files containing historical outage information
for excel_name in range(2015,2022):
    file_name = str(excel_name) + "_Annual_Summary.xls"
    df = pd.read_excel(file_name, skiprows=1, dtype=str) # read all the excel files
    for i in range(len(df)) :
        if (str(df.loc[i,"Number of Customers Affected"]).isnumeric() == False):
            #print(df.loc[i])
            df.loc[i,"Number of Customers Affected"] = 0
        states_impacted=[]
        # Check for multiple state names in a single column for an event type and divide them into multiple rows 
        for state in state_list: 
            if state in df.loc[i,"Area Affected"]:
                states_impacted.append(state)
        state_impacted_count = len(states_impacted)
        if (state_impacted_count != 0):
            customers_impacted = int(int(df.loc[i,"Number of Customers Affected"]) / state_impacted_count)
            for sc in range(0,state_impacted_count):
            #    df_new = df_new.append({'Month':df.loc[i,"Month"], 'Date Event Began':df.loc[i,"Date Event Began"], 'Time Event Began':df.loc[i,"Time Event Began"], 'Date of Restoration':df.loc[i,"Date of Restoration"], 'Time of Restoration':df.loc[i,"Time of Restoration"], 'Area Affected':states_impacted[sc], 'NERC Region':df.loc[i,"NERC Region"], 'Alert Criteria':df.loc[i,"Alert Criteria"], 'Event Type':df.loc[i,"Event Type"], 'Demand Loss (MW)':df.loc[i,"Demand Loss (MW)"], 'Number of Customers Affected': customers_impacted}, ignore_index=True)  
                 df_new_row = pd.DataFrame({'Month':[df.loc[i,"Month"]], 'Date Event Began':[df.loc[i,"Date Event Began"]], 'Time Event Began':[df.loc[i,"Time Event Began"]], 'Date of Restoration':[df.loc[i,"Date of Restoration"]], 'Time of Restoration':[df.loc[i,"Time of Restoration"]], 'Area Affected':[states_impacted[sc]], 'NERC Region':[df.loc[i,"NERC Region"]], 'Alert Criteria':[df.loc[i,"Alert Criteria"]], 'Event Type':[df.loc[i,"Event Type"]], 'Demand Loss (MW)':[df.loc[i,"Demand Loss (MW)"]], 'Number of Customers Affected': [customers_impacted]})
                 df_new = pd.concat([df_new, df_new_row])
# The unknown values were replaced by zero
df_new['Demand Loss (MW)'].replace('Unknown',0,inplace=True)
df_new['Time of Restoration'].replace('Unknown',np.nan,inplace=True)
df_new['Time Event Began'].replace('Unknown',np.nan,inplace=True)
df_new['Time of Restoration'].replace('Unknown ',np.nan,inplace=True)
df_new['Time Event Began'].replace('Unknown ',np.nan,inplace=True)
df_new.dropna(how='any', subset=['Time of Restoration', 'Time Event Began'], inplace=True)

df_new.reset_index(inplace=True)
#The next step was to fix all the dates in the dataset as they were not in a consistent format
FMD = '%m/%d/%Y'
FMT = '%Y-%m-%d'
FMT1 = '%H:%M:%S'
time_delta = []
delta_date=0
ref_date="12/31/2014" #use a reference date since the dates have many formats
for index, row in df_new.iterrows():
    start_date= row["Date Event Began"].split(" ")[0] #split the date to bring in correct format
    if (len(row["Date Event Began"].split(" "))>1): #substract the dates from the reference dates
        delta_start = (datetime.strptime(start_date, FMT) - datetime.strptime(ref_date, FMD)).total_seconds()
    else:
        delta_start = (datetime.strptime(start_date, FMD) - datetime.strptime(ref_date, FMD)).total_seconds()
    restore_date= row["Date of Restoration"].split(" ")[0]
    if (len(row["Date of Restoration"].split(" "))>1):
        delta_restore = (datetime.strptime(restore_date, FMT) - datetime.strptime(ref_date, FMD)).total_seconds()
    else:
        delta_restore = (datetime.strptime(restore_date, FMD) - datetime.strptime(ref_date, FMD)).total_seconds()
    delta_date = delta_restore - delta_start
    # calculate the total time required for restoration
    delta_time = (datetime.strptime(row["Time of Restoration"], FMT1) - datetime.strptime(row["Time Event Began"], FMT1)).total_seconds()
    restoration_time = delta_date + delta_time
    #One entry across the 7 years from Texas, has restore date in the past with respect to start date
    if (restoration_time < 0):
        restoration_time = 0
    time_delta.append((restoration_time)/3600)
    
df_new['tdelta'] = time_delta
#create a new column called figure of merit(fom) by multiplying the total time of restoration with number of customers affected
df_new['fom'] = df_new['tdelta'] * df_new['Number of Customers Affected']
df_new.to_csv('fom_output.csv')

# Data analysis - creation of power outage maps using figure of merit(fom) values

# read in a shapefile of US with states boundaries
us_states = gpd.read_file('us_states/us_states.shx')
#us_states.plot()
fig, ax = plt.subplots(1, 1)
us_states.plot(column ='state_name',ax=ax, legend=False)
fig.savefig('us_state.png')
plt.close(fig)

df_new['Area Affected'].unique
# us_states.head()
# us_states['state_name'].unique
#extract the year value and add it to a new column
df_new['year'] = pd.DatetimeIndex(df_new['Date Event Began']).year
df_new.to_csv('output_year.csv') 

#group by state and sum the fom column to find degree of outages for each state
df_grouped_sw = df_new.groupby(['Area Affected'], as_index=False).agg({"fom": "sum"})
df_grouped_sw=df_grouped_sw.rename(columns={"Area Affected" : "state_name"})
#merge the dataframe with the state shapefile
us_states_fom= us_states.merge(df_grouped_sw, on = 'state_name')
us_states_fom['fom_mil'] = round (us_states_fom['fom'] / 1000000 , 2 )

#ploting the power outage intensity based on states
vmin, vmax = 0, 10000
ax = us_states_fom.plot(column='fom_mil', cmap='tab20_r', scheme='quantiles')
# add colorbar
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(cmap='tab20_r')
#Geopanda bug in adding colorbar. Stack Overflow solution below used : https://stackoverflow.com/questions/36008648/colorbar-on-geopandas
sm._A = []
fig.colorbar(sm, cax=cax)
ax.set_title("Quantile data for FOM (Outage Duration x #Customers Affected )", fontsize=10)
fig.savefig('us_states_fom.png')
plt.close(fig)

# us_states_fom.to_csv('us_states_fom.csv')
#plot the outage intensity on yearly basis
for yr in range(2015,2022):
    df_year = df_new[df_new['year'] == yr]
    df_year=df_year.groupby(['Area Affected'], as_index=False).agg({"fom": "sum"})
    df_year=df_year.rename(columns={"Area Affected" : "state_name"})
    us_states_year = us_states.merge(df_year, on = 'state_name')
    
    ax = us_states_year.plot(column='fom', cmap='tab20_r', scheme='quantiles')
    fig = ax.get_figure()
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap='tab20_r')
    #Geopanda bug in adding colorbar. Stack Overflow solution below used : https://stackoverflow.com/questions/36008648/colorbar-on-geopandas
    sm._A = []
    fig.colorbar(sm, cax=cax)
    ax.set_title(("Quantiles for FOM (Outage Duration x #Customers Affected) - " + str(yr)), fontsize=10)
    us_states_fom.boundary.plot(ax=ax)
    fig.savefig(('us_states_fom_' + str(yr) + '.png'))
    plt.close(fig)

# import US shapefie with lat lon information 
us_states_lat_lon = gpd.read_file('s_22mr22/s_22mr22.shp')
#us_states_lat_lon.plot()

us_states_lat_lon=us_states_lat_lon.rename(columns={"NAME" : "state_name"})

df_grouped_sw.to_csv('grouped_sw_fom.csv')

#k-mean clustering performed to find clusters of outages in the US
#read in csvs to merge
csv1 = pd.read_csv('state_lat_lon.csv', skiprows=0, dtype=str)

csv1=csv1.rename(columns={"state" : "state_name"})

csv2 = pd.read_csv('grouped_sw_fom.csv', skiprows=0, dtype=str)

# merge the two csv files 
merged_lat_lon= csv1.merge(csv2, on = 'state_name')
new_lat_lon = merged_lat_lon[['state_name','latitude','longitude','fom']]

#Using the elbow method to find the optimal number of clusters
def opt_clus_num(data):
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init 
= 10, random_state = 0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("elbow.png")
    plt.close()
    #plt.show()
    

#opt_clus_num(new_lat_lon[['latitude', 'longitude','fom']]) # optimal number of clusters seen as 3

# clustering done using 3 and 5 clusters both 

#clustering using 5 clusters and using fom as a variable
kmeans = KMeans(n_clusters=5).fit(new_lat_lon[['latitude', 'longitude','fom']])
new_lat_lon['labels'] = kmeans.labels_
us_states = gpd.read_file('us_states/us_states.shx')
fom_kmean = us_states.merge(new_lat_lon, on = 'state_name')
fig, ax = plt.subplots(1, 1)
fom_kmean.plot(column ='labels',ax=ax, cmap='jet', legend=True)
fig.savefig('fom_kmean.png')


#import the substation shapefile 
us_power = gpd.read_file('Substations (1)/02993c2e-6b84-4322-9069-b66a1fbe0fb1.gdb')
us_states = us_states.to_crs('WGS 84')
us_power = us_power.to_crs('WGS 84')
state_power = gpd.sjoin(us_states, us_power)
state_substations = state_power['state_name'].value_counts()
state_substations.to_csv('sps.csv')


sps = pd.read_csv('sps.csv', skiprows=1, names=['state_name', 'substation_count'])
df_sps = merged_lat_lon.merge(sps, on = 'state_name')
#import the state demographics data also
dem = pd.read_csv('state_demographics.csv', skiprows=0)
dem = dem[['State','Population.2014 Population','Population.Population per Square Mile','Income.Per Capita Income']]
dem = dem.rename(columns={"State" : "state_name", "Population.2014 Population" : "population", "Population.Population per Square Mile" : "pop_den", "Income.Per Capita Income" : "per_capita_inc"})
#merge all the dataset
df_sps_pop = df_sps.merge(dem, on = 'state_name')
df_sps_pop.drop(['Unnamed: 0'], axis=1, inplace=True)
df_sps_pop_trim = df_sps_pop.drop(['state_name'], axis=1)


opt_clus_num(df_sps_pop_trim)
#opt_clus_num(new_lat_lon[['fom']])
#Clustering using  using 5 clusters and multiple variables
kmeans = KMeans(n_clusters=5).fit(df_sps_pop_trim)
df_sps_pop['labels'] = kmeans.labels_
us_states = gpd.read_file('us_states/us_states.shx')
fom_kmean = us_states.merge(df_sps_pop, on = 'state_name')
#us_states_fom.plot(column='fom', cmap='jet', scheme='quantiles', legend=False)
fig, ax = plt.subplots(1, 1)
fom_kmean.plot(column ='labels',ax=ax, cmap='plasma', legend=True)
ax.set_title("Similarity clusters based on power outage susceptibility)", fontsize=10)
fig.savefig('fom_kmean2.png')
plt.close(fig)

# ****After clustering creating, further deep diving into finding cause of outages was done. 
#  API calls created to find the main cause of the outages

df_top = pd.read_csv('output_year.csv', skiprows=0)
# finding the top 10 most severe outages in the US during the last 7 years
event_df_sorted = df_top.sort_values(by=['fom'], ascending=False).reset_index(drop=True)
event_df_sorted_10 = event_df_sorted.head(10)


event_df_sorted_1 = event_df_sorted_10[["Area Affected","fom"]]
#ax = event_df_sorted_1.plot(kind='barh', figsize=(8, 10), color='navy', zorder=2, width=0.85)

#bar graph showing the states impacted by the most severe outages in the last 7 years 
plt.style.use('fivethirtyeight')
#fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax = event_df_sorted_1.plot(kind='bar', figsize=(9, 14), color='navy', zorder=2, width=0.85)
ax.set_xticklabels(event_df_sorted_1['Area Affected'])
#plt.xticks(rotation = 38)
plt.title("Top 10 worst outage events")
plt.xlabel("State",fontsize=14)
plt.ylabel("FOM (Outage hours x Number of customers affected)")
plt.savefig('histogram_outage.png',bbox_inches="tight")
plt.close()


# read in the fips code (unique greographic codes for each state)
state_fips = pd.read_csv('state_fips.csv')
states_fips = state_fips.rename(columns={"st" : "Area Affected"})
event_df_sorted_10= event_df_sorted_10.merge(states_fips, on = 'Area Affected')
event_df_sorted_10 = event_df_sorted_10.sort_values(by=['fom'], ascending=False).reset_index(drop=True)

# extraxt information like start date , month and end date for the 10 major events to plot all the causes for all the 10 events 
for index, row in event_df_sorted_10.iterrows():
    state= row['Area Affected']
    print(state)
    state = state.replace(' ','+')
    fips = str(row['stusps'])
    event_date = row['Date Event Began']
    print(event_date)
    m_start = row['Date Event Began'].split("/")[0]
    d_start = row['Date Event Began'].split("/")[1]
    y_start = row['Date Event Began'].split("/")[2]
    m_end = row['Date of Restoration'].split("/")[0]
    d_end = row['Date of Restoration'].split("/")[1]
    y_end = row['Date of Restoration'].split("/")[2]
    #https://www.ncdc.noaa.gov/stormevents/csv?eventType=ALL&beginDate_mm=02&beginDate_dd=10&beginDate_yyyy=2021&endDate_mm=02&endDate_dd=23&endDate_yyyy=2021&county=ALL&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=48%2CTEXAS
    
    # call the url for NOAA data consisting of the event information 
    noaa_url = 'https://www.ncdc.noaa.gov/stormevents/csv?eventType=ALL&beginDate_mm=' + m_start + '&beginDate_dd=' + d_start + '&beginDate_yyyy=' + y_start + '&endDate_mm=' + m_end + '&endDate_dd=' + d_end + '&endDate_yyyy=' + y_end + '&county=ALL&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=' + fips + '%2C' + state.upper()
    df_noaa = pd.read_csv(noaa_url)
    df_noaa.groupby('EVENT_TYPE').size().plot.bar()
    plt.title("Weather event:" + state + " on " + event_date)
    #plt.tight_layout()
    #plt.xticks(rotation = 80)
    plt.savefig(('NOAA_' + event_date.replace("/","_") + state + '.png'),bbox_inches="tight")
    plt.close()
    
    #sleep(5)

#Extract Totao Interchange (MW) from EIA for an outage event
# Total interchange = Demand - Net Generation
# Extract start and end dates to make api call
state= event_df_sorted_10.head(1)['Area Affected'].values[0]
fips = str(event_df_sorted_10.head(1)['stusps'].values[0])
m_start = (event_df_sorted_10.head(1)['Date Event Began'].values[0]).split("/")[0]
d_start = (event_df_sorted_10.head(1)['Date Event Began'].values[0]).split("/")[1]
y_start = (event_df_sorted_10.head(1)['Date Event Began'].values[0]).split("/")[2]
m_end = (event_df_sorted_10.head(1)['Date of Restoration'].values[0]).split("/")[0]
d_end = (event_df_sorted_10.head(1)['Date of Restoration'].values[0]).split("/")[1]
y_end = (event_df_sorted_10.head(1)['Date of Restoration'].values[0]).split("/")[2]

# Files on EIA available at 6 months granularity. Process start month to choose file
if (int(m_start) < 7):
    month_string = 'Jan_Jun'
    counter = 0
else:
    month_string = 'Jul_Dec'
    counter = 6

#URL for EIA data
eia_url = 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_' + y_start + '_' + month_string + '.csv'
df_eia = pd.read_csv(eia_url)
df_eia_roi = df_eia[df_eia['Balancing Authority']=='ERCO']
list_x = list(range(0,len(df_eia_roi)))
#Read total interchange data from EIA
list_y = df_eia_roi[['Total Interchange (MW)']]
list_y = list_y.reset_index(drop=True)
list_ic = []
list_xm = []
#Data cleaning of interchange data, which has numbers with commas
for index, row in list_y.iterrows():
    interchange = row['Total Interchange (MW)']
    interchange = float(interchange.replace(',',''))
    list_ic.append(interchange)
    #Hourly x-axis converted to months. Added counter to handle months after June
    list_xm.append(index/(24*30) + counter)
plt.title("6 months data for " + state + " around " + event_date)
plt.xlabel("Months",fontsize=14)
plt.ylabel("Interchange (MW)")
plt.plot(list_xm,list_ic)
plt.savefig(('EIA_' + event_date.replace("/","_") + state + '.png'),bbox_inches="tight")
plt.close()
#Shape file output writing takes some time to run
fom_kmean.to_file('Output_outage_shp/Outage_cluster.shp')



















