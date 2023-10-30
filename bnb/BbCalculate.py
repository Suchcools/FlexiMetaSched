import numpy as np
import pandas as pd
import time
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore") 
num_jobs=0
job_Seq=[]
num_machines=0

last_node_dicts=[]
node_time_dicts=[]
node_dicts=[]

uptime=0
upSeq=[]
downtime=0
downSeq=[]

Job=None

orderid_Seq=[]

olen=75
""""seq":seqlist,"ftime":timelist,await:list"""
def read_data(filename: str):
    global Job,num_jobs,num_machines,job_Seq,orderid_Seq
    xls = pd.ExcelFile(filename)

    Job = pd.read_excel(xls, sheet_name='Job info')
    #Job["准备时间"]= pd.to_datetime(Job["准备时间"])
    #Job["到达时刻"]= pd.to_datetime(Job["到达时刻"])
    #Job["交货期"]= pd.to_datetime(Job["交货期"])
    Job["准备时间"]= Job["准备时间"].astype("float")
    Job["到达时刻"]= Job["到达时刻"].astype("float")
    Job["交货期"]= Job["交货期"].astype("float")
    #Job = Job.sort_values("交货期",ascending=True)
    #Job.reset_index(inplace=True)
    Job = Job.iloc[0:min(olen,len(Job))]
    Machine = pd.read_excel(xls, sheet_name='Machine info')
    
    PT1 = pd.read_excel(xls, sheet_name='Process 1 Time', index_col=0)
    PT1 = PT1.iloc[0:min(olen,len(PT1))]
    PT2 = pd.read_excel(xls, sheet_name='Process 2 Time', index_col=0)
    PT2 = PT2.iloc[0:min(olen,len(PT2))]

    num_machines=len(Machine)
    num_jobs=len(Job)
    job_Seq=Job.index
    orderid_Seq=np.linspace(0,num_jobs-1,num_jobs).astype("int")

    for i in range(len(PT1.columns)):
        Job.insert(0,"m1_%d" % i,np.array(PT1[PT1.columns[i]])*np.array(Job["需求量"]))
        Job.insert(1,"m2_%d" % i,np.array(PT2[PT2.columns[i]])*np.array(Job["需求量"]))
        Job.insert(1,"total_%d" % i,np.array(Job["准备时间"])+np.array(Job["m1_%d"%i])+np.array(Job["m2_%d"%i]))

    Job.insert(0,"sameid",np.zeros(num_jobs))
    Job=Job.to_dict(orient="dict")
    return Job, Machine, PT1, PT2

def init_step(istep):
    global last_node_dicts,node_dicts,num_machines,num_jobs,orderid_Seq,node_time_dicts
    node_dicts=[]
    node_time_dicts=[]
    downtime=0
    if istep==0:
        for io in range(len(orderid_Seq)):
            seqlist=[]
            sumtimelist=[]
            timelist=[]
            awaitlist=orderid_Seq.copy()
            sameid=Job["sameid"][awaitlist[io]]
            if sameid!=0:
                continue
            for j in range(num_machines):
                seqlist.append([])
                timelist.append([])
                sumtimelist.append(0)

            mtime= Job["m1_0"][awaitlist[io]]+Job["m2_0"][awaitlist[io]]
            m1time= Job["m1_0"][awaitlist[io]]
            m2time= Job["m2_0"][awaitlist[io]]
            mid=0
            for i in range(1,num_machines):
                if Job["m1_%d" % i][awaitlist[io]]+Job["m2_%d" % i][awaitlist[io]]<mtime:
                    mtime=Job["m1_%d" % i][awaitlist[io]]+Job["m2_%d" % i][awaitlist[io]]
                    m1time=Job["m1_%d" % i][awaitlist[io]]
                    m2time=Job["m2_%d" % i][awaitlist[io]]
                    mid=i
            seqlist[mid].append(awaitlist[io])
            awaitlist=np.delete(awaitlist,io)
            sumtimelist[mid]+=mtime
            timelist[mid].append(m1time)
            timelist[mid].append(m2time)
            node_dicts.append({"seq":seqlist, "tseq": timelist,"ftime":sumtimelist,"await":awaitlist,"suitNodes":1,"min":0})
            node_time_dicts.append(sumtimelist)
    else:
        ni=0
        for inode in last_node_dicts:
            for io in range(len(inode["await"])):
                #第四种模式
                """node4=inode.copy()
                sameid=Job.iloc[io]["sameid"]
                if sameid!=0:
                    for im in range(len(node4["seq"])):
                        if sameid in node4["seq"][im]:
                            node4["seq"][im].append(io)
                            node4["ftime"][im]+=Job.iloc[istep]["total_%d" % im]
                            if node4["ftime"][im]<=Job.iloc[istep]["交货期"]:
                                node_dicts.append(node4)"""
                
                #tempdict=Job.iloc[0:istep]
                #sleeptimelist=tempdict["准备时间"].to_list()
                #第三种模式
                for im in range(len(inode["seq"])):
                    for ii in range(len(inode["seq"][im])):
                        node3=deepcopy(inode)
                        if Job["m1_%d" % im][node3["await"][io]]<Job["准备时间"][node3["seq"][im][ii]] and Job["准备时间"][node3["await"][io]]>Job["m2_%d" % im][node3["seq"][im][ii]]:
                            node3["seq"][im].insert(ii+1,node3["await"][io])
                            node3["ftime"][im]+=Job["m1_%d" % im][node3["await"][io]]
                            if node3["ftime"][mid]<Job["交货期"][node3["await"][io]]:
                                node3["suitNodes"]+=1
                            node3["await"]=np.delete(node3["await"],io)
                            mintime=0
                            for subnode in range(len(inode["seq"])):
                                alltime=inode["ftime"][subnode]
                                for io0 in range(0,num_jobs):
                                    alltime+=Job["m1_%d" % subnode][io0]+Job["m2_%d" % subnode][io0]
                                if alltime>mintime:
                                    mintime=alltime
                            if node3["ftime"] not in node_time_dicts:
                                if downtime==0 or mintime<downtime:
                                    node_dicts=[node3]
                                    node_time_dicts=[node3["ftime"]]
                                    downtime=mintime
                                elif abs(mintime-downtime)<1e-6:
                                    node_dicts.append(node3)
                                    node_time_dicts.append(node3["ftime"])
                            else:
                                inid=node_time_dicts.index(node3["ftime"])
                                isnode=node_dicts[inid]
                                if node3["suitNodes"]>isnode["suitNodes"]:
                                    node_dicts[inid]=node3

                #第二种模式
                for im in range(len(inode["seq"])):
                    for ii in range(len(inode["seq"][im])):
                        node2=deepcopy(inode)
                        if Job["total_%d" % im][node2["await"][io]]<Job["准备时间"][node2["seq"][im][ii]]:
                            if node2["tseq"][im][2*ii+1]+Job["total_%d" % im][node2["await"][io]]<Job["交货期"][node2["await"][io]]:
                                node2["suitNodes"]+=1
                            node2["seq"][im].insert(ii+1,node2["await"][io])
                            node2["await"]=np.delete(node2["await"],io)
                            mintime=0
                            for subnode in range(len(inode["seq"])):
                                alltime=inode["ftime"][subnode]
                                for io0 in range(0,num_jobs):
                                    alltime+=Job["m1_%d" % subnode][io0]+Job["m2_%d" % subnode][io0]
                                if alltime>mintime:
                                    mintime=alltime
                            if node2["ftime"] not in node_time_dicts:
                                if downtime==0 or mintime<downtime:
                                    node_dicts=[node2]
                                    node_time_dicts=[node2["ftime"]]
                                    downtime=mintime
                                elif abs(mintime-downtime)<1e-6:
                                    node_dicts.append(node2)
                                    node_time_dicts.append(node2["ftime"])
                            else:
                                inid=node_time_dicts.index(node2["ftime"])
                                isnode=node_dicts[inid]
                                if node2["suitNodes"]>isnode["suitNodes"]:
                                    node_dicts[inid]=node2

                #第一种+第四种模式
                for mid in range(0,num_machines):
                    node1=deepcopy(inode)
                    node1["seq"][mid].append(node1["await"][io])
                    if node1["ftime"][mid]>=Job["到达时刻"][node1["await"][io]]:
                        node1["ftime"][mid]+=Job["total_%d" % im][node1["await"][io]]
                    else:
                        node1["ftime"][mid]= Job["到达时刻"][node1["await"][io]] + Job["total_%d" % im][node1["await"][io]]
                    if node1["ftime"][mid]<Job["交货期"][node1["await"][io]]:
                        node1["suitNodes"]+=1
                    node1["await"]=np.delete(node1["await"],io)
                    mintime=0
                    for subnode in range(len(node1["seq"])):
                        alltime=node1["ftime"][subnode]
                        for io0 in node1["await"]:
                            alltime+=Job["m1_%d" % subnode][io0]+Job["m2_%d" % subnode][io0]
                        if alltime>mintime:
                            mintime=alltime
                    if node1["ftime"] not in node_time_dicts:
                        if downtime==0 or mintime<downtime:
                            node_dicts=[node1]
                            node_time_dicts=[node1["ftime"]]
                            downtime=mintime
                        elif abs(mintime-downtime)<1e-6:
                            node_dicts.append(node1)
                            node_time_dicts.append(node1["ftime"])
                        #print("模式一",node1)
                    else:
                        inid=node_time_dicts.index(node1["ftime"])
                        isnode=node_dicts[inid]
                        if node1["suitNodes"]>isnode["suitNodes"]:
                            node_dicts[inid]=node1
            ni+=1
                     
    last_node_dicts=node_dicts.copy()

def cut_node():
    global last_node_dicts,node_dicts,num_machines,num_jobs,orderid_Seq,node_time_dicts
    tmp_timelist=[]
    tmp_nodelist=[]
    repetition_list=[]
    init_len=len(last_node_dicts)
    for ii in range(len(node_time_dicts)):
        if node_time_dicts[ii] not in tmp_timelist:
            tmp_timelist.append(node_time_dicts[ii])
            tmp_nodelist.append(last_node_dicts[ii])
            #print(last_node_dicts[ii])
        else:
            repetition_list.append(ii)
    if len(repetition_list)>0:
        for ri in range(len(repetition_list)-1,0):
            del last_node_dicts[repetition_list[ri]]  
    last_node_dicts=tmp_nodelist.copy()   
    res_len=len(last_node_dicts)   

def get_up():
    global uptime,upSeq,last_node_dicts,Job
    for inode in last_node_dicts:
        maxtime=0
        tempSeq_max=inode.copy()
        for subnode in range(len(inode["seq"])):
            alltime=inode["ftime"][subnode]
            if maxtime<alltime:
                maxtime=alltime
        if maxtime>uptime:
            uptime=maxtime
            upSeq=tempSeq_max.copy()
            
    #print(last_node_dicts)
 
def get_low():
    global downtime,downSeq,last_node_dicts,Job
    tempSeq_dict=[]
    node_dicts=[]
    downtime=0
    for inode in last_node_dicts:
        mintime=0
        for subnode in range(len(inode["seq"])):
            alltime=inode["ftime"][subnode]
            for io in range(0,num_jobs):
                alltime+=Job["m1_%d" % subnode][io]+Job["m2_%d" % subnode][io]
            if alltime>mintime:
                mintime=alltime
        if mintime<downtime or downtime==0:
            downtime=mintime
            tempSeq_dict=[{"node":inode,"time":mintime}]
        elif abs(mintime-downtime)<1e-6:
            tempSeq_dict.append({"node":inode,"time":mintime})

    node_dicts=[]
    for tnode in tempSeq_dict:
        if tnode["time"]<=downtime or abs(tnode["time"]-downtime)<1e-6:
            node_dicts.append(tnode["node"])

    last_node_dicts=node_dicts.copy()

def BnB(input,output):
    #读取数据
    read_data(input)
    for istep in range(num_jobs):
        init_step(istep)
        cut_node()
        #get_low()
        get_up()
    ResSeque=last_node_dicts[0]["seq"]
    total = []
    for i in range(len(ResSeque)):
        temp = []
        jo=0
        while jo <len(ResSeque[i])-1:
            #print("%d," % jo,end=" ")
            if Job["total_%d" % i][ResSeque[i][jo]]<Job["准备时间"][ResSeque[i][jo]]:
                temp.append("%d_0, %d_0, %d_1, %d_1," % (ResSeque[i][jo],ResSeque[i][jo+1],ResSeque[i][jo+1],ResSeque[i][jo]))
                jo+=2
            elif Job["m1_%d" % i][jo]<Job["准备时间"][ResSeque[i][jo]] and Job["准备时间"][jo]>Job["m2_%d" % i][jo]:
                temp.append("%d_0, %d_0, %d_1, %d_1," % (ResSeque[i][jo],ResSeque[i][jo+1],ResSeque[i][jo],ResSeque[i][jo+1]))
                jo+=2
            else:
                temp.append("%d_0, %d_1," % (ResSeque[i][jo],ResSeque[i][jo]))
                jo+=1
        if jo==len(ResSeque[i])-1:
            temp.append("%d_0, %d_1," % (ResSeque[i][jo],ResSeque[i][jo]))
        total.append([x.strip() for x in ''.join(temp).split(',')][:-1])
    # np.savez(output,data=ResSeque,order=total)