import os
import shutil
import sys
dir_path = os.path.dirname(sys.argv[0])
os.chdir(dir_path)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from functools import reduce
from shutil import rmtree
from multiprocessing import Pool
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
plt.rc('ytick', labelsize=7)

def granger(df,col1,col2,maxlag=5):
    x = grangercausalitytests(df[[col1, col2]].diff().dropna(),maxlag=maxlag,verbose=False)  # null hypoposis col2 does not granger cause col1
    lags = list(range(1,maxlag+1))
    lag_pv = np.array([x[lag][0]['ssr_chi2test'][1] for lag in lags])
    best_pv = min(lag_pv)
    try:
        best_lag = np.array(lags)[lag_pv == best_pv] if len(lag_pv == best_pv) == 1 else np.array(lags)[lag_pv == best_pv][0]
    except:
        best_lag = 0
    return([best_lag,best_pv])
def granger_mat(df,maxlag = 5):
    '''
    col1 is row col2 is columns
    each cell represents the best result for the test that the column does not granger cause the row
    if rejected then: col --granger cause-- row
    '''
    cols = df.columns
    mat = [[granger(df,col1,col2,maxlag) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat, columns=cols, index=cols)
    return(out_df)
def entropy(Y):
    """
    Also known as Shanon Entropy
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en
def jEntropy(Y,X):
    """
    H(Y;X)
    """
    YX = np.c_[Y,X]
    return entropy(YX)
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y,X)
def Theils_U(Y,X):
    '''
    :return: U(Y|X) the uncertenty of Y given X
    '''
    return(gain(Y,X)/entropy(Y))
def Theils_U_matrix(df):
    #calculate Theils U for each set of 2 columns in the df
    cols = df.columns
    mat = [[Theils_U(df[col1], df[col2]) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat,columns=cols,index=cols)
    return(out_df)
def transfer_time_to_s(x):
    # transdorms time column to seconds
    split_time = [float(y) for y in x.split(':')]
    time_in_s = np.multiply(split_time,[3600,60,1]).sum()
    return(time_in_s)
def cramer_v(x,y):
    #calculated cramers V for 2 arrays
    df_c = pd.DataFrame({"x": x, "y": y})
    cont_table = pd.crosstab(df_c["x"],df_c["y"])
    chi = chi2_contingency(cont_table,correction=False)[0]
    n = cont_table.sum().sum()
    v = np.sqrt((chi/n))
    return(v)
def cramer_v_matrix(df):
    #calculate cramers V for each set of 2 columns in the df
    cols = df.columns
    mat = [[cramer_v(df[col1], df[col2]) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat,columns=cols,index=cols)
    return(out_df)
def trim_by_time(df):
    #trims the data from s_time to e_time
    s_time = df["s_time"][df["action"] == 'start-end'].values[0]+0.001
    end_time = df["e_time"][df["action"] == 'start-end'].values[1]-0.001
    return(df[(df["s_time"] >= s_time) & (df["e_time"] <= end_time)])
def ct_helper(df1,df2,sec,order):
    #counts if utterance at df2 started 1 sec after df1 starts or {sec} sec after it ended
    s_times = []
    e_times = []
    t_times = []
    for index, row in df1.iterrows():
        interaction_df = df2[(df2["s_time"]>= row["s_time"] +1)&(df2["s_time"]<= row["e_time"] +sec)]
        if len(interaction_df) == 0:
            continue
        s_times.append(row["s_time"])
        e_times.append(interaction_df["e_time"].values[0])
        t_times.append(interaction_df["e_time"].values[0]-row["s_time"])
    df_out = pd.DataFrame({"action":["Conversational turns"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":[order]*len(s_times)})
    return(df_out)
def Conversational_turns(df,sec = 5):
    # creates Conversational turns data
    parent_utterance_df = df[df["action"] == 'Parent utterance']
    child_utterance_df = df[df["action"] == 'Child utterance']
    PC_frame = ct_helper(parent_utterance_df,child_utterance_df,sec,"PC")
    CP_frame = ct_helper(child_utterance_df,parent_utterance_df, sec, "CP")
    PCP_frame = ct_helper(PC_frame,CP_frame,sec,"PCP")
    CPC_frame = ct_helper(CP_frame,PC_frame, sec, "CPC")
    return(PC_frame,CP_frame,PCP_frame,CPC_frame)
def joint_attention(df,rap=True):
    #creates joint attention data
    #rap specifies to treat robots and props as both robots and props,
    #meaning that if the child gazed at robot and parent at robots and props it count as joint attention
    #counts if gaze at df2 started while df1 was gazing

    if rap:
        #deletes robot or prop and adds them as two seperate rows
        rap_df = df[df["sub_action"] == 'robot or prop']
        r_df = rap_df.copy()
        r_df["sub_action"] = "robot"
        p_df = rap_df.copy()
        p_df["sub_action"] = "props"
        df = pd.concat([df[(df["sub_action"] != 'robot or prop')],r_df,p_df]).reset_index(drop = True)

    parent_gaze_df = df[df["action"] == 'Parent gaze']
    child_gaze_df = df[df["action"] == 'Child gaze']
    s_times = []
    e_times = []
    t_times = []
    sub_action_gaze=[]
    #parent starts gaze
    for index, row in parent_gaze_df.iterrows():
        v1 = (child_gaze_df["s_time"] >= row["s_time"]) & (child_gaze_df["s_time"] <= row["e_time"]) #child gazed after parent
        v2 = (child_gaze_df["s_time"] <= row["s_time"]) & (child_gaze_df["e_time"] >= row["s_time"]) #parent gazed after child
        interaction_df = child_gaze_df[v1 | v2]
        interaction_df = interaction_df[interaction_df["sub_action"] == row["sub_action"]]
        if len(interaction_df) == 0:
            continue
        for ind, inter_row in interaction_df.iterrows():
            s_time = max(inter_row["s_time"],row["s_time"])
            e_time = min(inter_row["e_time"],row["e_time"])
            s_times.append(s_time)
            e_times.append(e_time)
            t_times.append(e_time-s_time)
            sub_action_gaze.append(row["sub_action"])

    df_out = pd.DataFrame({"action":["Joint attention"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":sub_action_gaze})
    return(df_out)
def mutual_gaze(df):
    parent_gaze_df = df[df["action"] == 'Parent gaze']
    parent_gaze_at_child = parent_gaze_df[parent_gaze_df["sub_action"]=="child"]
    child_gaze_df = df[df["action"] == 'Child gaze']
    child_gaze_at_parent = child_gaze_df[child_gaze_df["sub_action"]=="parent"]

    s_times = []
    e_times = []
    t_times = []
    for index, row in parent_gaze_at_child.iterrows():
        v1 = (child_gaze_at_parent["s_time"] >= row["s_time"]) & (child_gaze_at_parent["s_time"] <= row["e_time"]) #child gazed after parent
        v2 = (child_gaze_at_parent["s_time"] <= row["s_time"]) & (child_gaze_at_parent["e_time"] >= row["s_time"]) #parent gazed after child
        interaction_df = child_gaze_at_parent[v1 | v2]
        if len(interaction_df) == 0:
            continue
        for ind, inter_row in interaction_df.iterrows():
            s_time = max(inter_row["s_time"],row["s_time"])
            e_time = min(inter_row["e_time"],row["e_time"])
            s_times.append(s_time)
            e_times.append(e_time)
            t_times.append(e_time-s_time)

    df_out = pd.DataFrame({"action":["Mutual gaze"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":"MG"})
    return(df_out)
def fillnas(df):
    action = df["action"]
    who = df["who"]
    action_first_word = [act.split(" ")[0] for act in action]
    action_first_word = ["Parent" if (act == "Verbal") else act for act in action_first_word]
    who_filled = [w if isinstance(w,str) else act for w,act in zip(who,action_first_word)]
    df["who"] = who_filled

    sub = df["sub_action"]
    action_last_words = [reduce(lambda x, y: x + " " + y, act.split(" ")[1:]) if
                         len(act.split(" ")[1:]) > 0 else [" "] for act in action]
    sub_filled = [s if isinstance(s, str) else word for s, word in zip(sub, action_last_words)]
    df["sub_action"] = sub_filled
    return(df)
def df_preprocess(df):
    #give names to columns
    df.columns =["action","who","s_time","e_time","t_time","sub_action"]
    df = df.convert_dtypes()  #converts to string
    #convert to time objects
    for time_col in ["s_time","e_time","t_time"]:
        df[time_col] = df[time_col].apply(transfer_time_to_s)

    #df = trim_by_time(df) #trims data
    df = fillnas(df)# fill nas
    PC_frame, CP_frame, PCP_frame, CPC_frame = Conversational_turns(df,sec = 5) #makes conversational turns
    ja_frame = joint_attention(df) #makes joint attentions
    mg_frame = mutual_gaze(df) # makes mutual gaze
    df = pd.concat([df,PC_frame, CP_frame, PCP_frame, CPC_frame,ja_frame,mg_frame]).reset_index(drop=True)
    df["action:sub_action"] = df["action"] + ":" + df["sub_action"].fillna("")
    return(df)
# def transform_to_time_representation_old(df,col = "action",time_stamp_jumps=1):
#     #return the df in the time domain
#     end_of_vid = ceil(df["e_time"].values[-1])
#     start_of_vid = floor(df["s_time"].values[0])
#     time_indicates = list(range(start_of_vid, end_of_vid, time_stamp_jumps))
#     # for each observation an array of times it appears in
#     times= pd.Series([list(range(floor(row["s_time"]), ceil(row["e_time"]), time_stamp_jumps)) for index,row in df.iterrows()],
#                      name = "times")
#     features = df[col].unique()
#     feaueres_times = {feature: np.concatenate(times[df[col] == feature].reset_index(drop=True)) for feature in features} # for each feature the times it appears in
#     feaueres_time_binary = {feature:[1 if t in feaueres_times[feature] else 0 for t in time_indicates] for feature in features} # for each feature does it appear in time i
#     feaueres_time_binary["time"] = time_indicates
#     out_df = pd.DataFrame(feaueres_time_binary)
#     return(out_df)
def make_crosstab(df,col):
    #makes a crosstab for spesific column
    time_sums = pd.crosstab(df[col],"sum",values = df["t_time"],aggfunc = 'sum') #sums time for each action
    time_counts = pd.crosstab(df[col], "count", values=df["t_time"],aggfunc='count')  # count number of events for each action
    interaction_length = df["e_time"].max()-df["s_time"].min()
    time_sums_normalize = time_sums/interaction_length
    time_counts_normalize = time_counts / interaction_length
    out_df = pd.concat([time_sums,time_counts,time_sums_normalize,time_counts_normalize],axis= 1)
    out_df.columns = ["total time","count","normalized total time","normalized count"]
    return(out_df)

def time_window(df,col_base,act_base,col_count,act_count,s_w,e_w):
    '''
    df = dataframe not in time domain
    col_base = takes window for this column
    act_base = takes window for this action(or sub or action:sub)
    col_count = counts the # of events for this column
    act_count = counts the # of events for this action(or sub or action:sub)
    s_w = how long (in second) to start window after col_base
    e_w = how long to end the window after col_base
    return: dict of the actions and it's respective histogram
    '''
    action_e_times = df[df[col_base]==act_base]["e_time"]
    counter_s_times = df[df[col_count]==act_count]["s_time"]

    # in_time_frame = lambda s_t,e_t: e_t + s_w <= s_t <= e_t + e_w
    # time_count = {e_time: sum([in_time_frame(s_time,e_time) for s_time in counter_s_times]) for e_time in action_e_times}
    # hist = np.histogram(list(time_count.values()), bins=list(range(s_w,max(list(time_count.values()))+2)))

    bins = np.array(range(s_w, e_w))
    hist_vals = []
    for bin, next_bin in zip(bins, bins[1:]):
        hist_vals.append(sum(sum(
            [(e_time + bin <= counter_s_times) & (counter_s_times < e_time + next_bin) for e_time in action_e_times])))
    hist = (np.array(hist_vals),bins)


    return({f"{col_base}&{act_base}&{col_count}&{act_count}":hist})
def all_windows(df,s_w,e_w):
    cols = ["action","sub_action","action:sub_action"]
    col_uniques = np.array([f"{col}&{p_a}" for col in cols for p_a in np.unique(df[col])])
    dict = {}
    [dict.update(time_window(df,col_act_base.split("&")[0],col_act_base.split("&")[1],col_act_count.split("&")[0],col_act_count.split("&")[1],s_w,e_w))
     for col_act_base in col_uniques for col_act_count in col_uniques[col_uniques !=col_act_base]]
    out_df = pd.DataFrame({"col base":[],
                           "action base":[],
                           "col count":[],
                           "action count":[],
                           "hist":[],
                           "bins" :[]})
    split_loc = lambda x,i: x.split("&")[i]
    for key,item in dict.items():
        out_df = out_df.append({"col base":split_loc(key,0),
                           "action base":split_loc(key,1),
                           "col count":split_loc(key,2),
                           "action count":split_loc(key,3),
                           "hist":item[0],
                            "bins":item[1]},ignore_index=True)
    return(out_df)


def transform_to_time_representation(df,col = "action",time_stamp_jumps=1):
    # return the df in the time domain
    if "start-end" in df["action"]:
        end_of_vid = df[df["sub_action"] == "end"]["e_time"].values[0]
        start_of_vid = df[df["sub_action"] == "start"]["s_time"].values[0]
        df = df[df["action"] != "start-end"]
    else:
        end_of_vid = df["e_time"].max()
        start_of_vid = df["s_time"].min()

    time_indicates = np.arange(start_of_vid, end_of_vid, time_stamp_jumps)
    # for each observation an array of times it appears in
    # times= pd.Series([np.arange(row["s_time"], row["e_time"], time_stamp_jumps) for index,row in df.iterrows()],
    #                  name = "times")
    times = pd.Series([(row["s_time"], row["e_time"]) for index,row in df.iterrows()],
                     name = "times")
    features = df[col].unique()
    feaueres_times = {feature: times[df[col] == feature].reset_index(drop=True) for feature in features} # for each feature the times it appears in
    feaueres_time_binary = {feature:[1 if sum([1 if tup[0]-time_stamp_jumps <= t < tup[1] else 0 for tup in feaueres_times[feature]]) >0 else 0
                                     for t in time_indicates]
                            for feature in features} # for each feature does it appear in time i
    feaueres_time_binary["time"] = time_indicates
    out_df = pd.DataFrame(feaueres_time_binary)
    return(out_df)

def time_window_hist(df,col_base,action_base,col_count,action_count = 'all',save_path = ''):
    '''
    :param df: dataframe
    :param col_base: [action,sub_action,action:sub_action] for action base
    :param action_base: action to count window for
    :param col_count: [action,sub_action,action:sub_action] for action count
    :param action_count: action to count in window
    :return: dataframe containing histograms
    '''
    df = df[df["action base"] == action_base]
    df = df[df["col base"] == col_base]
    df = df[df["col count"] == col_count]
    if not isinstance(action_count,str):
        l1 = df["action count"].apply(lambda x: x in action_count)
        df = df[l1]
    elif action_count != "all":
        l1 = df["action count"].apply(lambda x: x == action_count)
        df = df[l1]
    df.reset_index(inplace = True)
    hist = df["hist"]
    bins = df["bins"]
    action_count = df["action count"]
    if isinstance(hist[0],str):
        make_array = lambda x: [int(y)  for y in x.strip("]").strip("[").split(" ") if y != '']
        hist = hist.apply(make_array)
        bins = bins.apply(make_array)
    x_axis = np.unique([y for x in bins.values for y in x])
    f = {}
    f[col_count] = []
    f.update({x:[] for x in x_axis})
    out_df = pd.DataFrame(f)
    for h,bin,act in zip (hist,bins,action_count):
        d = {}
        d[col_count] = act
        d.update(dict(zip(bin[:-1],h)))
        out_df= out_df.append(d,ignore_index=True)
    out_df.fillna(0,inplace=True)
    out_df.index = out_df[col_count].apply(lambda x: x.replace(" ","\n"))
    out_df.drop(col_count, axis=1,inplace=True)

    if save_path != '':
        fig = sns.heatmap(out_df, annot=True, linewidths=.5)
        fig.set_title(action_base)
        plt.savefig(f"{save_path} {col_base} window hist.png")
        plt.show()
        plt.close()

    return(out_df)

def process_data(file,col_base="action", action_base = "Child gaze", col_count = "action", action_count = 'all',time_stamp_jumps = 0.5,overwrite = False):
    file_base = file[:-4]
    path = os.path.join("output",file_base)
    # goes to next file if the file was already processed
    if os.path.exists(path):
        print(f"{file_base} already in output files")
        if not overwrite:
            return
        shutil.rmtree(path)
    os.makedirs(path)

    file_path = os.path.join("files",file)
    df = pd.read_csv(file_path, sep='\t', engine='python',header = None)
    df = df_preprocess(df)
    df.to_csv(os.path.join(path,f"{file_base}.csv"))
    print(f"made csv for {file_base}")
    pd.concat([make_crosstab(df,"action"),make_crosstab(df,"sub_action"),make_crosstab(df,"action:sub_action")]).to_csv(os.path.join(path,f"{file_base} sum_count.csv"))

    df_time_action = transform_to_time_representation(df,"action",time_stamp_jumps)
    df_time_sub_action = transform_to_time_representation(df, "sub_action", time_stamp_jumps)
    df_time_sub_action_sub_action = transform_to_time_representation(df, "action:sub_action", time_stamp_jumps)
    path_file_base = os.path.join(path,file_base)
    df_time_action.to_csv(f"{path_file_base} action time rep.csv")
    df_time_sub_action.to_csv(f"{path_file_base} sub_action time rep.csv")
    df_time_sub_action_sub_action.to_csv(f"{path_file_base} action_sub_action time rep.csv")
    print(f"made time representation for {file_base}")
    all_windows_df = all_windows(df,-3,5)
    all_windows_df.to_csv(f"{path_file_base} windows.csv")
    print(f"made windows for {file_base}")
    Theils_U_matrix(df_time_action).to_csv(f"{path_file_base} action_U.csv")
    Theils_U_matrix(df_time_sub_action).to_csv(f"{path_file_base} sub_action_U.csv")
    Theils_U_matrix(df_time_sub_action_sub_action).to_csv(f"{path_file_base} action_sub_action _U.csv")
    print(f"made Theils_U_matrix for {file_base}")
    granger_mat(df_time_action.drop("time",axis = 1),5).to_csv(f"{path_file_base} granger action.csv")
    granger_mat(df_time_sub_action.drop("time",axis = 1), 5).to_csv(f"{path_file_base} granger sub_action.csv")
    granger_mat(df_time_sub_action_sub_action.drop("time",axis = 1), 5).to_csv(f"{path_file_base} granger action_sub_action.csv")
    print(f"made granger for {file_base}")
    time_hist = time_window_hist(all_windows_df,col_base, action_base, col_count, action_count,path_file_base)
    print(f"made time window for {file_base}")

def stich_frames(df_array,stiching_with = np.nan,stich_len = 10):
    """
    df_array: array of dataframes to concat
    stiching_with: value to stich with
    stich_len = length of stiching
    """

    cols = np.unique(sum([list(t_df.columns) for t_df in df_array],[]))
    d = {}
    for col in cols:
        df_col = []
        for df in df_array:
            if col in df.columns:
                if sum(col == df.columns) > 1:
                    df_col.extend(df[col].iloc[:, 0])
                else:
                    df_col.extend(list(df[col].values))
            else:
                df_col.extend([stiching_with]*len(df))
        d[col] = df_col
    # for i,df_t in enumerate(df_array):
    #     for col in cols:
    #         n = len(df_t)
    #         if col not in df_t.columns:
    #             df_t[col] = [stiching_with]* n
    #     df_array[i] = df_t[cols].reset_index(drop=True).loc[:,~df_t.columns.duplicated()]
    # df = df_array[0]
    # for df_to_add in df_array[1:]:
    #     df = pd.concat([df,pd.DataFrame({col:[stiching_with]*stich_len for col in cols}),df_to_add]).reset_index(drop=True)
    return(pd.DataFrame(d))

def make_df_dict(features):
    #file discreptions
    files_disc = {file:{"lesson":file[:2],"type":file[2],"participent":file[-3:]} for file in os.listdir("output")}
    #all participants numbers
    unique_part = np.unique([file[-3:] for file in os.listdir("output")])
    #dictenary with values for each participent
    base_dict = {"participent":[]}
    first_robot_dict = {f"{feature}_first_robot":[] for feature in features}
    second_robot_dict = {f"{feature}_second_robot": [] for feature in features}
    tablet_dict = {f"{feature}_tablet": [] for feature in features}
    base_dict.update(first_robot_dict)
    base_dict.update(second_robot_dict)
    base_dict.update(tablet_dict)
    df_dict = {participent:base_dict.copy() for participent in unique_part}
    for participent in unique_part:
        df_dict[participent]["participent"] = participent
    return (df_dict,files_disc,unique_part,base_dict)

def fill_dict(files_disc,df_dict):
    '''
    fills dictinory for t-test analysis
    '''
    for file in files_disc:
        path_to_sum_count = os.path.join("output",file,f"{file} sum_count.csv")
        temp_df = pd.read_csv(path_to_sum_count)
        type = files_disc[file]["type"]
        participent = files_disc[file]["participent"]
        lesson = files_disc[file]["lesson"]
        # add for tablet type
        if type == "t":
            for feature in features:
                if feature not in ["Child gaze:props_robot_tablet","Parent gaze:props_robot_tablet"]:
                    try:
                        df_dict[participent][f"{feature}_tablet"] = temp_df[temp_df["Unnamed: 0"] == feature]["normalized total time"].values[0]
                    except:
                        df_dict[participent][f"{feature}_tablet"] = 0
                else:
                    if feature == "Child gaze:props_robot_tablet":
                        try:
                            df_dict[participent][f"{feature}_tablet"] = temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0]-temp_df[temp_df["Unnamed: 0"] == "Child gaze:parent"]["normalized total time"].values[0]
                        except:
                            df_dict[participent][f"{feature}_tablet"] = temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0]
                    else:
                        try:
                            df_dict[participent][f"{feature}_tablet"] = temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0]-temp_df[temp_df["Unnamed: 0"] == "Parent gaze:child"]["normalized total time"].values[0]
                        except:
                            try:
                                df_dict[participent][f"{feature}_tablet"] = temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}_tablet"] = 0
        else:
            if lesson == "l1":
                for feature in features:
                    if feature not in ["Child gaze:props_robot_tablet", "Parent gaze:props_robot_tablet"]:
                        try:
                            df_dict[participent][f"{feature}_first_robot"] = \
                            temp_df[temp_df["Unnamed: 0"] == feature]["normalized total time"].values[0]
                        except:
                            df_dict[participent][f"{feature}_first_robot"] = 0
                    else:
                        if feature == "Child gaze:props_robot_tablet":
                            try:
                                df_dict[participent][f"{feature}_first_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze:parent"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}_first_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0]
                        else:
                            try:
                                df_dict[participent][f"{feature}_first_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze:child"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}_first_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0]
            elif lesson == "l3":
                for feature in features:
                    if feature not in ["Child gaze:props_robot_tablet", "Parent gaze:props_robot_tablet"]:
                        try:
                            df_dict[participent][f"{feature}_second_robot"] = \
                            temp_df[temp_df["Unnamed: 0"] == feature]["normalized total time"].values[0]
                        except:
                            df_dict[participent][f"{feature}_second_robot"] = 0
                    else:
                        if feature == "Child gaze:props_robot_tablet":
                            try:
                                df_dict[participent][f"{feature}_second_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze:parent"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}_second_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0]
                        else:
                            try:
                                df_dict[participent][f"{feature}_second_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze:child"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}_second_robot"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0]
            else:
                lessons =[]
                types = []
                for file_2 in files_disc:
                    if files_disc[file_2]["participent"] == participent:
                        lessons.append(files_disc[file_2]["lesson"])
                        types.append(files_disc[file_2]["type"])
                try:
                    if np.array(types)[np.array(lessons) == "l1"][0] =="t":
                        ending = "_first_robot"
                    else:
                        ending = "_second_robot"
                except:
                    #if we got here there is probbalby missing sessions
                    ending = "_first_robot"
                for feature in features:
                    if feature not in ["Child gaze:props_robot_tablet", "Parent gaze:props_robot_tablet"]:
                        try:
                            df_dict[participent][f"{feature}{ending}"] = \
                            temp_df[temp_df["Unnamed: 0"] == feature]["normalized total time"].values[0]
                        except:
                            df_dict[participent][f"{feature}{ending}"] = 0
                    else:
                        if feature == "Child gaze:props_robot_tablet":
                            try:
                                df_dict[participent][f"{feature}{ending}"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze:parent"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}{ending}"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Child gaze"]["normalized total time"].values[0]
                        else:
                            try:
                                df_dict[participent][f"{feature}{ending}"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0] - \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze:child"]["normalized total time"].values[0]
                            except:
                                df_dict[participent][f"{feature}{ending}"] = \
                                temp_df[temp_df["Unnamed: 0"] == "Parent gaze"]["normalized total time"].values[0]
    return(df_dict)

if __name__ == '__main__':
    '''
    actions = ['Child utterance', 'Child gaze', 'Child gesture',
       'Parent utterance', 'Verbal scaffolding', 'Parent gaze',
       'Parent gesture', 'Child prop manipulation', 'Parent prop manipulation',
       'Conversational turns', 'Joint attention', 'Mutual gaze', 'time',
       'Child affect', 'Parent affect', 'Parent affective touch',
       'Child affective touch', 'Non-verbal scaffolding']
    '''
    process = False
    multi_process = False
    #load data
    files = os.listdir("files")
    if process:
        #multiprocess the data
        if multi_process:
            #change number of workers according to the available number
            p = Pool(8)
            with p:
                p.map(process_data,files)
        else:
            for file in files:
                process_data(file)

    processed_files = os.listdir("output")
    # stich dataframes
    df_array = [pd.concat([pd.read_csv(os.path.join("output",file,f"{file} action time rep.csv")),
                                    pd.read_csv(os.path.join("output",file,f"{file} sub_action time rep.csv")),
                                    pd.read_csv(os.path.join("output",file,f"{file} action_sub_action time rep.csv"))], axis=1)
                                    for file in processed_files]
    robot_files_bool = [file[2] == "r" for file in processed_files]
    tablet_files_bool = [file[2] == "t" for file in processed_files]
    df = stich_frames(df_array)
    df_r = stich_frames(np.array(df_array)[robot_files_bool])
    df_t = stich_frames(np.array(df_array)[tablet_files_bool])

    #specific t-tests
    features = ['Child gesture','Conversational turns','Mutual gaze','Parent gesture',"Child gaze:parent",
                "Child gaze:props_robot_tablet","Parent gaze:props_robot_tablet","Parent gaze:child","Verbal scaffolding:affective","Verbal scaffolding:cognitive",
                "Verbal scaffolding:technical","Parent affective touch","Child affect","Parent affect"]
    df_dict,files_disc,unique_part,base_dict = make_df_dict(features)
    df_dict = fill_dict(files_disc, df_dict)
    #make dataframe for t test
    t_test_df = pd.DataFrame(base_dict)
    for participent in unique_part:
        t_test_df = t_test_df.append(df_dict[participent], ignore_index=True)
    t_test_df.to_csv(os.path.join("analysis output","t_test_df.csv"))

    ttest_rel
    t_results = pd.DataFrame({"feature":[],
                              "t_score":[],
                              "pv":[],
                              "mean_robot":[],
                              "std_robot":[],
                              "mean_tablet":[],
                              "std_tablet":[]})

    for feat in features:
        robot_feat = f"{feat}_first_robot"
        tablet_feat = f"{feat}_tablet"
        robot_vals = t_test_df[robot_feat]
        tablet_vals = t_test_df[tablet_feat]
        l1 = [t != [] for t in robot_vals]
        l2 = [t != [] for t in tablet_vals]
        valid = [l_1 and l_2 for l_1,l_2 in zip(l1,l2)]
        if sum(valid) > 0:
            valid_robot = robot_vals[valid]
            valid_tablet = tablet_vals[valid]
            t_test = ttest_rel(valid_robot,valid_tablet,alternative = "two-sided")
            t_results = t_results.append({"feature": feat,
             "t_score": t_test[0],
             "pv": t_test[1],
             "mean_robot": valid_robot.mean(),
             "std_robot": valid_robot.std(),
             "mean_tablet": valid_tablet.mean(),
             "std_tablet": valid_tablet.std()}, ignore_index=True)
    t_results = t_results.dropna().reset_index()
    # benjamini hocberg transformation
    t_results["bh_pv"] = multipletests(t_results["pv"], alpha=0.05, method="fdr_bh")[1]

    print(t_results)

    #granger features
    col1 = ['robot pointing','robot pointing','robot pointing','robot pointing','robot pointing',
            'robot text:positive feedback','robot text:positive feedback','robot text:positive feedback',
            "robot text:pick up","robot text:pick up","robot text:pick up","robot text:pick up","robot text:pick up",
            "robot text:pick up","robot text:pick up","robot text:pick up","robot text:pick up",
            "robot text:pick up","robot text:pick up","robot text:pick up","robot text:pick up"]
    col2 = ['Child gaze:props','Child gaze:robot','Parent gaze:props','Parent gaze:robot','Joint attention',
            'Parent affective scaffolding','Parent affective touch','Child affective touch',
            'Child gaze:props','Child gaze:robot','Parent gaze:props','Parent gaze:robot',"Mutual gaze",
            'Child utterance',"Parent utterance","Conversational turns","Verbal scaffolding","Child gesture:point at prop",
            "Parent gesture:point at prop","Child prop manipulation","Parent prop manipulation"]
    lags = []
    pv = []
    df_r['Parent affective scaffolding'] = [max(x,y) for x,y in zip(df_r["Non-verbal scaffolding:affective"],df["Verbal scaffolding:affective"])]
    for col_1,col_2 in zip(col1,col2):
        #specific granger analysis
        granger_res = granger(df_r, col_1, col_2, maxlag=5)
        lags.append(granger_res[0])
        pv.append(granger_res[1])
    granger_df = pd.DataFrame({"col1":col1,
                               "col2":col2,
                               "lag" :lags,
                               "pv":pv})
    granger_df["bh_pv"] = multipletests(granger_df["pv"], alpha=0.05, method="fdr_bh")[1]

    bh_pv_full = multipletests(pd.concat([t_results["pv"],granger_df["pv"]]), alpha=0.05, method="fdr_bh")[1]
    t_results["bh_pv_full"] = bh_pv_full[:len(t_results)]
    granger_df["bh_pv_full"] = bh_pv_full[-len(granger_df):]
    t_results.to_csv(os.path.join("analysis output", "t_results_df.csv"))
    granger_df.to_csv(os.path.join("analysis output","granger_df.csv"))
    print(granger_df)


    # for i, df in enumerate([df_time_action, df_time_sub_action, df_time_sub_action_sub_action]):
    #     sns.heatmap(df.corr(), annot=True)
    #     plt.savefig(f"fig_{i}.png")
    #     plt.show()




