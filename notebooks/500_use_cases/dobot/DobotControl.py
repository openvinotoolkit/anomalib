import threading
import DobotDllType as dType

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#将dll读取到内存中并获取对应的CDLL实例
#Load Dll and get the CDLL object
api = dType.load()
#建立与dobot的连接
#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])

if (state == dType.DobotConnect.DobotConnect_NoError):
    
    #清空队列
    #Clean Command Queued
    dType.SetQueuedCmdClear(api)
    
    #设置运动参数
    #Async Motion Params Setting
    dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)

    #回零
    #Async Home
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    #设置ptpcmd内容并将命令发送给dobot
    #Async PTP Motion
    for i in range(0, 5):
        if i % 2 == 0:
            offset = 50
        else:
            offset = -50
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200 + offset, offset, offset, offset, isQueued = 1)[0]

    #开始执行指令队列
    #Start to Execute Command Queue
    dType.SetQueuedCmdStartExec(api)

    #如果还未完成指令队列则等待
    #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)

    #停止执行指令
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)

#断开连接
#Disconnect Dobot
dType.DisconnectDobot(api)
