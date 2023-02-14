from ctypes import *
import time,  platform
import os

def enum(**enums):
    return type("Enum", (), enums)

EndType = enum(EndTypeCustom=0, 
    EndTypeSuctionCup=1, 
    EndTypeGripper=2, 
    EndTypeLaser=3,
    EndTypePen = 4,  
    EndTypeMax=5)

DevType = enum(Idle=0,
               Conntroller=1,
               Magician=2,
               MagicianLite=3
               )

ParamsMode = enum(JOG=0,
                other=1)



class DevInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("devId", c_int),
        ("type", c_int),
        ("firmwareName", c_byte * 50),
        ("firwareVersion", c_byte * 50),
        ("runTime", c_float)
               ]

class ConnectInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("masterDevInfo", DevInfo),
        ("slaveDevInfo1", DevInfo),
        ("slaveDevInfo2", DevInfo)
    ]

class UpgradeFWReadyCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("fwSize", c_uint32),
        ("md5", c_char_p)
    ]

class DeviceID(Structure):
    _pack_ = 1
    _fields_ = [
        ("deviceID1", c_uint32),
        ("deviceID2", c_uint32),
        ("deviceID3", c_uint32)
    ]

masterId = 0
slaveId = 0

class DeviceVersion(Structure):
    _pack_ = 1
    _fields_ = [
        ("fw_majorVersion", c_byte),
        ("fw_minorVersion", c_byte),
        ("fw_revision", c_byte),
        ("fw_alphaVersion", c_byte),
        ("hw_majorVersion", c_byte),
        ("hw_minorVersion", c_byte),
        ("hw_revision", c_byte),
        ("hw_alphaVersion", c_byte)
    ]

# For EndTypeParams
class EndTypeParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("xBias", c_float),
        ("yBias", c_float),
        ("zBias", c_float)
        ]

class Pose(Structure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("rHead", c_float),
        ("joint1Angle", c_float),
        ("joint2Angle", c_float),
        ("joint3Angle", c_float),
        ("joint4Angle", c_float)
        ]

class Kinematics(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocity", c_float),
        ("acceleration", c_float)
        ]

class AlarmsState(Structure):
    _pack_ = 1
    _fields_ = [
        ("alarmsState", c_int32)
        ]

class HOMEParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float), 
        ("y", c_float), 
        ("z", c_float), 
        ("r", c_float)
        ]

class HOMECmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("temp", c_float)
        ]
        
class AutoLevelingCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("controlFlag", c_ubyte),
        ("precision", c_float)
        ]
        
class EMotor(Structure):
    _pack_ = 1
    _fields_ = [
        ("index", c_byte), 
        ("isEnabled", c_byte), 
        ("speed", c_int32)
        ]
        
class EMotorS(Structure):
    _pack_ = 1
    _fields_ = [
        ("index", c_byte), 
        ("isEnabled", c_byte), 
        ("speed", c_int32), 
        ("distance", c_uint32)
        ]
        
##################  Arm orientation定义   ##################
ArmOrientation = enum(
    LeftyArmOrientation=0, 
    RightyArmOrientation=1)
    
##################  点动示教部分   ##################

class JOGJointParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("joint1Velocity", c_float), 
        ("joint2Velocity", c_float), 
        ("joint3Velocity", c_float), 
        ("joint4Velocity", c_float), 
        ("joint1Acceleration", c_float),
        ("joint2Acceleration", c_float),
        ("joint3Acceleration", c_float),
        ("joint4Acceleration", c_float)
        ]

class JOGCoordinateParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("xVelocity", c_float), 
        ("yVelocity", c_float), 
        ("zVelocity", c_float), 
        ("rVelocity", c_float), 
        ("xAcceleration", c_float),
        ("yAcceleration", c_float),
        ("zAcceleration", c_float),
        ("rAcceleration", c_float)
        ]

class JOGCommonParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocityRatio", c_float), 
        ("accelerationRatio", c_float)
        ]

class JOGLParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocity",  c_float), 
        ("acceleration",  c_float)
    ]


JC = enum(JogIdle=0, 
    JogAPPressed=1, 
    JogANPressed=2, 
    JogBPPressed=3, 
    JogBNPressed=4,
    JogCPPressed=5,
    JogCNPressed=6,
    JogDPPressed=7,
    JogDNPressed=8,
    JogEPPressed=9,
    JogENPressed=10)

class JOGCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("isJoint", c_byte), 
        ("cmd", c_byte)
        ]

##################  再现运动部分   ##################

class PTPJointParams(Structure):
    _fields_ = [
        ("joint1Velocity", c_float), 
        ("joint2Velocity", c_float), 
        ("joint3Velocity", c_float), 
        ("joint4Velocity", c_float), 
        ("joint1Acceleration", c_float),
        ("joint2Acceleration", c_float),
        ("joint3Acceleration", c_float),
        ("joint4Acceleration", c_float)
        ]
        
class PTPCoordinateParams(Structure):
    _fields_ = [
        ("xyzVelocity", c_float), 
        ("rVelocity", c_float),
        ("xyzAcceleration", c_float), 
        ("rAcceleration", c_float)
        ]

class PTPLParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocity",  c_float), 
        ("acceleration",  c_float)
    ]

class PTPJumpParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("jumpHeight", c_float), 
        ("zLimit", c_float)
        ]

class PTPCommonParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocityRatio", c_float), 
        ("accelerationRatio", c_float)
        ]

PTPMode = enum(
    PTPJUMPXYZMode=0,
    PTPMOVJXYZMode=1,
    PTPMOVLXYZMode=2,
    
    PTPJUMPANGLEMode=3,
    PTPMOVJANGLEMode=4,
    PTPMOVLANGLEMode=5,
    
    PTPMOVJANGLEINCMode=6,
    PTPMOVLXYZINCMode=7, 
    PTPMOVJXYZINCMode=8, 
    
    PTPJUMPMOVLXYZMode=9)

InputPin = enum( InputPinNone=0,
    InputPin1=1,
    InputPin2=2,
    InputPin3=3,
    InputPin4=4,
    InputPin5=5,
    InputPin6=6,
    InputPin7=7,
    InputPin8=8)

InputLevel = enum(InputLevelBoth=0,
    InputLevelLow=1,
    InputLevelHigh=2)

OutputPin = enum(
    SIGNALS_O1=1,
    SIGNALS_O2=2,
    SIGNALS_O3=3,
    SIGNALS_O4=4,
    SIGNALS_O5=5,
    SIGNALS_O6=6,
    SIGNALS_O7=7,
    SIGNALS_O8=8)

class PTPCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("ptpMode", c_byte),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("rHead", c_float)
        ]
        
class DeviceCountInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("deviceRunTime",  c_uint64),
        ("devicePowerOn",  c_uint32),
        ("devicePowerOff", c_uint32)
        ]
        
class PTPWithLCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("ptpMode", c_byte),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("rHead", c_float),
        ("l", c_float)
        ]

##################  Continuous path   ##################

class CPParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("planAcc", c_float),
        ("juncitionVel", c_float),
        ("acc", c_float), 
        ("realTimeTrack",  c_byte)
        ]

ContinuousPathMode = enum(
    CPRelativeMode=0,
    CPAbsoluteMode=1)

class CPCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("cpMode", c_byte),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("velocity", c_float)
        ]

class CP2Cmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("cpMode", c_byte),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("velocity", c_float)
        ]
        
class CPCommonParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocityRatio", c_float), 
        ("accelerationRatio", c_float)
        ]

##################  圆弧：ARC   ##################
class ARCPoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("rHead", c_float)
    ]
        
class ARCParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("xyzVelocity", c_float), 
        ("rVelocity", c_float),
        ("xyzAcceleration", c_float), 
        ("rAcceleration", c_float)
        ]

class ARCCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("cirPoint", ARCPoint),
        ("toPoint", ARCPoint)
    ]
    
class CircleCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("cirPoint", ARCPoint),
        ("toPoint", ARCPoint)
    ]

class ARCCommonParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("velocityRatio", c_float), 
        ("accelerationRatio", c_float)
        ]

##################  User parameters   ##################

class WAITParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("unitType", c_byte)
        ]

class WAITCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("waitTime", c_uint32)
        ]

TRIGMode = enum(
    TRIGInputIOMode = 0,
    TRIGADCMode=1)
    
TRIGInputIOCondition = enum(
    TRIGInputIOEqual = 0,
    TRIGInputIONotEqual=1)
    
TRIGADCCondition = enum(
    TRIGADCLT = 0,
    TRIGADCLE=1, 
    TRIGADCGE = 2,
    TRIGADCGT=3)
    
class TRIGCmd(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("mode", c_byte), 
        ("condition",  c_byte), 
        ("threshold", c_uint16)
        ]

GPIOType = enum(
    GPIOTypeDummy = 0, 
    GPIOTypeDO = 1,
    GPIOTypePWM=2,
    GPIOTypeDI=3, 
    GPIOTypeADC=4, 
    GPIOTypeDIPU=5, 
    GPIOTypeDIPD=6)
    
class IOMultiplexing(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("multiplex", c_byte)
        ]
        
class IODO(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("level", c_byte)
        ]
        
class IOPWM(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("frequency", c_float), 
        ("dutyCycle", c_float)
        ]
        
class IODI(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("level", c_byte)
        ]
        
class IOADC(Structure):
    _pack_ = 1
    _fields_ = [
        ("address", c_byte), 
        ("value", c_int)
        ]

class UserParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("params1", c_float),
        ("params2", c_float),
        ("params3", c_float),
        ("params4", c_float),
        ("params5", c_float),
        ("params6", c_float),
        ("params7", c_float),
        ("params8", c_float)
        ]

ZDFCalibStatus = enum(
    ZDFCalibNotFinished=0,
    ZDFCalibFinished=1)
    

class WIFIIPAddress(Structure):
    _pack_ = 1
    _fields_ = [
        ("dhcp", c_byte),
        ("addr1", c_byte),
        ("addr2", c_byte),
        ("addr3", c_byte),
        ("addr4", c_byte),
        ]
        
class WIFINetmask(Structure):
    _pack_ = 1
    _fields_ = [
        ("addr1", c_byte),
        ("addr2", c_byte),
        ("addr3", c_byte),
        ("addr4", c_byte),
        ]
        
class WIFIGateway(Structure):
    _pack_ = 1
    _fields_ = [
        ("addr1", c_byte),
        ("addr2", c_byte),
        ("addr3", c_byte),
        ("addr4", c_byte),
        ]
        
class WIFIDNS(Structure):
    _pack_ = 1
    _fields_ = [
        ("addr1", c_byte),
        ("addr2", c_byte),
        ("addr3", c_byte),
        ("addr4", c_byte),
        ]

ColorPort = enum(
    PORT_GP1 = 0, 
    PORT_GP2 = 1,
    PORT_GP4 = 2,
    PORT_GP5 = 3
    )
    
InfraredPort = enum(
    PORT_GP1 = 0, 
    PORT_GP2 = 1,
    PORT_GP4 = 2,
    PORT_GP5 = 3
    )
    
UART4PeripheralsType = enum(
    UART4PeripheralsUART = 0,
    UART4PeripheralsWIFI = 1,
    UART4PeripheralsBLE = 2,
    UART4PeripheralsCH375 = 3
    )
##################  API result   ##################

DobotConnect = enum(
    DobotConnect_NoError=0,
    DobotConnect_NotFound=1,
    DobotConnect_Occupied=2)

DobotCommunicate = enum(
    DobotCommunicate_NoError=0,
    DobotCommunicate_BufferFull=1,
    DobotCommunicate_Timeout=2,
    DobotCommunicate_InvalidParams=3,
    DobotCommunicate_InvalidDevice=4
    )

isUsingLinearRail = False
##################  API func   ##################

#parker add 2018 8 29 添加Wifi设置模块退出标志位
QuitDobotApiFlag = True

def load():
    if platform.system() == "Windows":
        print("您用的dll是64位，为了顺利运行，请保证您的python环境也是64位")
        print("python环境是：",platform.architecture())
        return CDLL("./DobotDll.dll",  RTLD_GLOBAL)
    elif platform.system() == "Darwin":
        return CDLL("./libDobotDll.dylib",  RTLD_GLOBAL)
    elif platform.system() == "Linux":
        return cdll.loadLibrary("libDobotDll.so")


def dSleep(ms):
    time.sleep(ms / 1000)  

def gettime():
    return [time.time()]


def SetDebugEnable(api, flag=False):
    result = api.SetDebugEnable(flag)


def SearchDobot(api,  maxLen=1000):
    szPara = create_string_buffer(1000) #((len(str(maxLen)) + 4) * maxLen + 10)
    l = api.SearchDobot(szPara,  maxLen)
    if l == 0:
        return []
    ret = szPara.value.decode("utf-8") 
    
    def fix(devices):
        for index in range(len(devices)):
            device = devices[index]
            if "(" in device and ")" not in device:
                yield device + " " + devices[index + 1]
            elif "(" not in device and ")" in device:
                pass
            else:
                yield device
        
    return list(fix(ret.split(" ")))
    
masterId = 0
slaveId = 0
masterDevType = 0
slaveDevType = 0


def ConnectDobot(api, portName, baudrate):
    global masterId, slaveId, masterDevType, slaveDevType

    szPara = create_string_buffer(100)
    szPara.raw = portName.encode("utf-8") 
    connectInfo = ConnectInfo()

    result = api.ConnectDobot(szPara, baudrate, byref(connectInfo))
    if result != DobotConnect.DobotConnect_NoError:
        return [result, 0, 0, 0, 0, 0, 0, 0]
    masterId = connectInfo.masterDevInfo.devId
    masterDevType = connectInfo.masterDevInfo.type
    try:
        if masterDevType == DevType.Conntroller:
            if connectInfo.slaveDevInfo1.type == 0 and connectInfo.slaveDevInfo2.type == 0:
                slaveId = -1
                slaveDevType = 0
                try:
                    fwName = str(connectInfo.masterDevInfo.firmwareName, encoding="utf-8").strip(b'\x00'.decode())
                    fwVer = str(connectInfo.masterDevInfo.firwareVersion, encoding="utf-8").strip(b'\x00'.decode())
                    # print("masterId: ", masterId, connectInfo.slaveDevInfo1.devId, connectInfo.slaveDevInfo2.devId, fwName, fwVer)
                except Exception as e:
                    print(e)
            else:
                slaveId = connectInfo.slaveDevInfo1.devId if connectInfo.slaveDevInfo1.type != DevType.Idle else connectInfo.slaveDevInfo2.devId
                fwName = str(connectInfo.slaveDevInfo1.firmwareName, encoding="utf-8").strip(b'\x00'.decode()) if connectInfo.slaveDevInfo1.type != DevType.Idle else str(connectInfo.slaveDevInfo2.firmwareName, encoding="utf-8").strip(b'\x00'.decode())
                fwVer = str(connectInfo.slaveDevInfo1.firwareVersion, encoding="utf-8").strip(b'\x00'.decode()) if connectInfo.slaveDevInfo1.type != DevType.Idle else str(connectInfo.slaveDevInfo2.firwareVersion, encoding="utf-8").strip(b'\x00'.decode())
                slaveDevType = connectInfo.slaveDevInfo1.type if connectInfo.slaveDevInfo1.type != DevType.Idle else connectInfo.slaveDevInfo2.type
                # slaveDevType = dType.DevType.MagicianLite  # for test
        else:
            slaveId = 0
            slaveDevType = 0
            fwName = str(connectInfo.masterDevInfo.firmwareName, encoding="utf-8").strip(b'\x00'.decode())
            fwVer = str(connectInfo.masterDevInfo.firwareVersion, encoding="utf-8").strip(b'\x00'.decode())

    except Exception as e:
        print(e)
    return [result, masterDevType, slaveDevType, fwName, fwVer, masterId, slaveId, connectInfo.masterDevInfo.runTime]


def DisconnectDobot(api):
    api.DisconnectDobot(c_int(masterId))


def GetMarlinVersion(api):
    api.GetMarlinVersion(c_int(masterId), c_int(slaveId))


def PeriodicTask(api):
    api.PeriodicTask()


def SetCmdTimeout(api, times):
    api.SetCmdTimeout(c_int(masterId), times)



def DobotExec(api):
    return [api.DobotExec()]


def GetQueuedCmdCurrentIndex(api):
    queuedCmdIndex = c_uint64(0)
    queuedCmdIndex1 = c_uint64(0)
    if masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        # if isUsingLinearRail:
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle: 
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    else:
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    return [queuedCmdIndex.value, queuedCmdIndex1.value]


def GetQueuedCmdMotionFinish(api):
    isFinish = c_bool(False)
    while(True):
        result = api.GetQueuedCmdMotionFinish(c_int(masterId), c_int(slaveId),byref(isFinish))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(2)
            continue
        break

    if isFinish.value != None:
        return [isFinish.value]
    else:
        return [False]


def SetQueuedCmdStartExec(api):
    # 特殊处理
    if slaveDevType == DevType.Magician:
        while (True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while (True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while (True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        while(True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while (True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break



def SetQueuedCmdStopExec(api):
    # 滑轨特殊处理
    if slaveDevType == DevType.Magician:
        while (True):
            result = api.SetQueuedCmdStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while (True):
            result = api.SetQueuedCmdStopExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while (True):
            result = api.SetQueuedCmdStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        while(True):
            result = api.SetQueuedCmdStartExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while (True):
            result = api.SetQueuedCmdStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break

       
 
def SetQueuedCmdForceStopExec(api):
    # 滑轨特殊处理
    if slaveDevType == DevType.Magician:
        while (True):
            result = api.SetQueuedCmdForceStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while (True):
            result = api.SetQueuedCmdForceStopExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while (True):
            result = api.SetQueuedCmdForceStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        while(True):
            result = api.SetQueuedCmdForceStopExec(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while (True):
            result = api.SetQueuedCmdForceStopExec(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break

    

def SetQueuedCmdStartDownload(api,  totalLoop, linePerLoop):
    while(True):
        result = api.SetQueuedCmdStartDownload(c_int(masterId), c_int(slaveId), totalLoop, linePerLoop)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def SetQueuedCmdStopDownload(api):
    while(True):
        result = api.SetQueuedCmdStopDownload(c_int(masterId), c_int(slaveId))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def SetQueuedCmdClear(api):
    # 滑轨特殊处理
    # return [api.SetQueuedCmdClear(c_int(masterId), c_int(slaveId))]
    if slaveDevType == DevType.Magician:
        while(True):
            result = api.SetQueuedCmdClear(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while (True):
            result = api.SetQueuedCmdClear(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while (True):
            result = api.SetQueuedCmdClear(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        while(True):
            result = api.SetQueuedCmdClear(c_int(masterId), c_int(-1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while (True):
            result = api.SetQueuedCmdClear(c_int(masterId), c_int(slaveId))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    return [result]


def SetDeviceSN(api, str): 
    szPara = create_string_buffer(25)
    szPara.raw = str.encode("utf-8")
    while(True):
        result = api.SetDeviceSN(c_int(masterId), c_int(slaveId), szPara)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetDeviceSN(api): 
    szPara = create_string_buffer(25)
    while(True):
        result = api.GetDeviceSN(c_int(masterId), c_int(slaveId), szPara,  25)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    ret = szPara.value.decode("utf-8") 
    return [ret]


def SetDeviceName(api, str):
    szPara = create_string_buffer(len(str) * 4)
    szPara.raw = str.encode("utf-8")
    while(True):
        result = api.SetDeviceName(c_int(masterId), c_int(slaveId), szPara)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def SetDeviceNumName(api, num): 
    cNum = c_int(num)
    while(True):
        result = api.SetDeviceName(c_int(masterId), c_int(slaveId), cNum)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetDeviceName(api): 
    szPara = create_string_buffer(66)
    while(True):
        result = api.GetDeviceName(c_int(masterId), c_int(slaveId), szPara,  100)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    ret = szPara.value.decode("utf-8")
    return [ret]
    

def GetDeviceVersion(api):
    deviceVersion = DeviceVersion()
    if (masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle)):
        while(True):
            result = api.GetDeviceVersion(c_int(masterId), c_int(-1), byref(deviceVersion))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        return [deviceVersion.fw_majorVersion, deviceVersion.fw_minorVersion, deviceVersion.fw_revision, deviceVersion.fw_alphaVersion,
            deviceVersion.hw_majorVersion, deviceVersion.hw_minorVersion, deviceVersion.hw_revision, deviceVersion.hw_alphaVersion]
    elif masterDevType == DevType.MagicianLite:
        while(True):
            result = api.GetDeviceVersion(c_int(masterId), c_int(slaveId), byref(deviceVersion))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        return [deviceVersion.fw_majorVersion, deviceVersion.fw_minorVersion, deviceVersion.fw_revision, deviceVersion.fw_alphaVersion,
            deviceVersion.hw_majorVersion, deviceVersion.hw_minorVersion, deviceVersion.hw_revision, deviceVersion.hw_alphaVersion]

    elif masterDevType == DevType.Magician:
        while(True):
            result = api.GetDeviceVersion(c_int(masterId), c_int(slaveId), byref(deviceVersion))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        return [deviceVersion.fw_majorVersion, deviceVersion.fw_minorVersion, deviceVersion.fw_revision, deviceVersion.fw_alphaVersion]


def SetDeviceWithL(api, isWithL, version=0, isQueued=0):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    queuedCmdIndex = c_uint64(0)
    while(True):
        print(tempSlaveId)
        result = api.SetDeviceWithL(c_int(masterId), c_int(tempSlaveId), c_bool(isWithL), c_uint8(version), c_bool(isQueued), byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetDeviceWithL(api):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    isWithL = c_bool(False)
    while(True):
        result = api.GetDeviceWithL(c_int(masterId), c_int(tempSlaveId), byref(isWithL))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isWithL.value]


def GetDeviceTime(api):
    time = c_uint32(0)
    while(True):
        result = api.GetDeviceTime(c_int(masterId), c_int(slaveId), byref(time))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [time.value]


def GetDeviceID(api):
    deviceID = DeviceID()
    CommunicateCount = 0
    timeout = False
    while(True):
        result = api.GetDeviceID(c_int(masterId), c_int(-1), byref(deviceID))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            if CommunicateCount > 3:
                timeout = True
                break
            else:
                CommunicateCount += 1
                dSleep(5)
                continue
        timeout = False
        break
    if timeout:
        return [result, 0, 0, 0]
    else:
        return [result, deviceID.deviceID1, deviceID.deviceID2, deviceID.deviceID3]


def GetDeviceInfo(api):
    info = DeviceCountInfo()
    while(True):
        result = api.GetDeviceInfo(c_int(masterId), c_int(slaveId), byref(info))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [info.deviceRunTime, info.devicePowerOn, info.devicePowerOff]


def ResetPose(api, manual, rearArmAngle, frontArmAngle):
    c_rearArmAngle = c_float(rearArmAngle)
    c_frontArmAngle = c_float(frontArmAngle)
    while(True):
        result = api.ResetPose(c_int(masterId), c_int(slaveId), manual, c_rearArmAngle, c_frontArmAngle)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetPose(api):
    pose = Pose()
    while(True):
        result = api.GetPose(c_int(masterId), c_int(slaveId), byref(pose))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pose.x, pose.y, pose.z,pose.rHead, pose.joint1Angle, pose.joint2Angle, pose.joint3Angle, pose.joint4Angle]


def GetPoseL(api):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    l = c_float(0)
    while(True):
        result = api.GetPoseL(c_int(masterId), c_int(tempSlaveId), byref(l))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    #parker add 20190524  判断返回的值是否为空
    if not math.isnan(l.value):
        return [l.value]
    else:
        return [0.0]


def GetKinematics(api):
    kinematics = Kinematics()
    while(True):
        result = api.GetKinematics(c_int(masterId), c_int(slaveId), byref(kinematics))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [kinematics.velocity, kinematics.acceleration]


def GetAlarmsState(api,  maxLen=1000):
    alarmsState = create_string_buffer(maxLen) 
    #alarmsState = c_byte(0)
    len = c_int(0)
    while(True):
        result = api.GetAlarmsState(c_int(masterId), c_int(slaveId), alarmsState, byref(len),  maxLen)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [alarmsState.raw, len.value]
    

def ClearAllAlarmsState(api):
    while(True):
        result = api.ClearAllAlarmsState(c_int(masterId), c_int(slaveId))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetUserParams(api):
    param = UserParams()
    while(True):
        result = api.GetUserParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.params1,param.params2,param.params3,param.params4,param.params5,param.params6,param.params7,param.params8]


def SetHOMEParams(api,  x,  y,  z,  r,  isQueued=0):
    param = HOMEParams()
    param.x = x
    param.y = y
    param.z = z
    param.r = r
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetHOMEParams(c_int(masterId), c_int(slaveId), byref(param),  isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetHOMEParams(api):
    param = HOMEParams()
    while(True):
        result = api.GetHOMEParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.x, param.y, param.z, param.r]


def SetHOMECmd(api, temp, isQueued=0):
    cmd = HOMECmd()
    cmd.temp = temp
    queuedCmdIndex = c_uint64(0)
    queuedCmdIndex1 = c_uint64(0)
    # 滑轨的特殊处理
    if masterDevType == DevType.Magician:
        # 只有Magician
        while(True):
            result = api.SetHOMECmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        # 外部控制器加MagicianLite
        # if isUsingLinearRail:#如果使用了滑轨，发给控制盒
        while(True):
            result = api.SetHOMECmd(c_int(masterId), c_int(-1), byref(cmd), isQueued, byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while(True):
            result = api.SetHOMECmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        # 外部控制器
        # if isUsingLinearRail:
        while(True):
            result = api.SetHOMECmd(c_int(masterId), c_int(-1), byref(cmd), isQueued, byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        # 其他情况
        while(True):
            result = api.SetHOMECmd(c_int(masterId), c_int(slaveDevType), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break

    return [queuedCmdIndex.value, queuedCmdIndex1.value]
    

def SetAutoLevelingCmd(api, controlFlag, precision, isQueued=0):
    cmd = AutoLevelingCmd()
    cmd.controlFlag = controlFlag
    cmd.precision = precision
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetAutoLevelingCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetAutoLevelingResult(api):
    precision = c_float(0)
    while(True):
        result = api.GetAutoLevelingResult(c_int(masterId), c_int(slaveId), byref(precision))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [precision.value]


def SetArmOrientation(api,  armOrientation, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetArmOrientation(c_int(masterId), c_int(slaveId), armOrientation, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def GetArmOrientation(api):
    armOrientation = c_int32(0)
    while(True):
        result = api.GetArmOrientation(c_int(masterId), c_int(slaveId), byref(armOrientation))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [armOrientation.value]
    

def SetHHTTrigMode(api, hhtTrigMode):
    while(True):
        result = api.SetHHTTrigMode(c_int(masterId), c_int(slaveId), hhtTrigMode)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetHHTTrigMode(api):
    hhtTrigMode = c_int(0)
    while(True):
        result = api.GetHHTTrigMode(c_int(masterId), c_int(slaveId), byref(hhtTrigMode))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [hhtTrigMode.value]


def SetHHTTrigOutputEnabled(api, isEnabled):
    while(True):
        result = api.SetHHTTrigOutputEnabled(c_int(masterId), c_int(slaveId), isEnabled)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetHHTTrigOutputEnabled(api):
    isEnabled = c_int32(0)
    while(True):
        result = api.GetHHTTrigOutputEnabled(c_int(masterId), c_int(slaveId), byref(isEnabled))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isEnabled.value]


def GetHHTTrigOutput(api):
    isAvailable = c_int32(0)
    result = api.GetHHTTrigOutput(c_int(masterId), c_int(slaveId), byref(isAvailable))
    if result != DobotCommunicate.DobotCommunicate_NoError or isAvailable.value == 0:
        return [False]
    return [True]

   

def SetEndEffectorParams(api, xBias, yBias, zBias, isQueued=0):
    param = EndTypeParams()
    param.xBias = xBias
    param.yBias = yBias
    param.zBias = zBias
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetEndEffectorParams(c_int(masterId), c_int(slaveId), byref(param),  isQueued,  byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
        

def GetEndEffectorParams(api):
    param = EndTypeParams()
    while(True):
        result = api.GetEndEffectorParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.xBias, param.yBias, param.zBias]
    

def SetEndEffectorLaser(api, enableCtrl,  on, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetEndEffectorLaser(c_int(masterId), c_int(slaveId), enableCtrl,  on,  isQueued,  byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
        

def GetEndEffectorLaser(api):
    isCtrlEnabled = c_int(0)
    isOn = c_int(0)
    while(True):
        result = api.GetEndEffectorLaser(c_int(masterId), c_int(slaveId), byref(isCtrlEnabled),  byref(isOn))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isCtrlEnabled.value, isOn.value]
    

def SetEndEffectorSuctionCup(api, enableCtrl,  on, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetEndEffectorSuctionCup(c_int(masterId), c_int(slaveId), enableCtrl,  on,  isQueued,  byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
        

def GetEndEffectorSuctionCup(api):
    enableCtrl = c_int(0)
    isOn = c_int(0)
    while(True):
        result = api.GetEndEffectorSuctionCup(c_int(masterId), c_int(slaveId), byref(enableCtrl),  byref(isOn))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isOn.value]
    

def SetEndEffectorGripper(api, enableCtrl,  on, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetEndEffectorGripper(c_int(masterId), c_int(slaveId), enableCtrl,  on,  isQueued,  byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
        

def GetEndEffectorGripper(api):
    enableCtrl = c_int(0)
    isOn = c_int(0)
    while(True):
        result = api.GetEndEffectorGripper(c_int(masterId), c_int(slaveId), byref(enableCtrl),  byref(isOn))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isOn.value]


def SetJOGJointParams(api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration, j4Velocity, j4Acceleration, isQueued=0):
    jogParam = JOGJointParams()
    jogParam.joint1Velocity = j1Velocity
    jogParam.joint1Acceleration = j1Acceleration
    jogParam.joint2Velocity = j2Velocity
    jogParam.joint2Acceleration = j2Acceleration
    jogParam.joint3Velocity = j3Velocity
    jogParam.joint3Acceleration = j3Acceleration
    jogParam.joint4Velocity = j4Velocity
    jogParam.joint4Acceleration = j4Acceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetJOGJointParams(c_int(masterId), c_int(slaveId), byref(jogParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetJOGJointParams(api):
    param = JOGJointParams()
    while(True):
        result = api.GetJOGJointParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.joint1Velocity, param.joint1Acceleration, param.joint2Velocity, param.joint2Acceleration, param.joint3Velocity, param.joint3Acceleration, param.joint4Velocity, param.joint4Acceleration]


def SetJOGCoordinateParams(api, xVelocity, xAcceleration, yVelocity, yAcceleration, zVelocity, zAcceleration, rVelocity, rAcceleration, isQueued=0):
    param = JOGCoordinateParams()
    param.xVelocity = xVelocity
    param.xAcceleration = xAcceleration
    param.yVelocity = yVelocity
    param.yAcceleration = yAcceleration
    param.zVelocity = zVelocity
    param.zAcceleration = zAcceleration
    param.rVelocity = rVelocity
    param.rAcceleration = rAcceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetJOGCoordinateParams(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetJOGCoordinateParams(api):
    param = JOGCoordinateParams()
    while(True):
        result = api.GetJOGCoordinateParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.xVelocity, param.xAcceleration, param.yVelocity, param.yVelocity, param.zVelocity, param.zAcceleration, param.rVelocity, param.rAcceleration]


def SetJOGLParams(api, velocity, acceleration, isQueued=0):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    param = JOGLParams()
    param.velocity = velocity
    param.acceleration = acceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetJOGLParams(c_int(masterId), c_int(tempSlaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def GetJOGLParams(api):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    param = JOGLParams()
    while(True):
        result = api.GetJOGLParams(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.velocity,  param.acceleration]


def SetJOGCommonParams(api, value_velocityratio, value_accelerationratio, isQueued=0):
    param = JOGCommonParams()
    param.velocityRatio = value_velocityratio
    param.accelerationRatio = value_accelerationratio
    queuedCmdIndex = c_uint64(0)

    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        while(True):
            result = api.SetJOGCommonParams(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while(True):
            result = api.SetJOGCommonParams(c_int(masterId), c_int(-1), byref(param), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while(True):
            result = api.SetJOGCommonParams(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle:
        while(True):
            result = api.SetJOGCommonParams(c_int(masterId), c_int(-1), byref(param), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while(True):
            result = api.SetJOGCommonParams(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break

    return [queuedCmdIndex.value]


def GetJOGCommonParams(api):
    param = JOGCommonParams()
    while(True):
        result = api.GetJOGCommonParams(c_int(masterId), c_int(slaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.velocityRatio, param.accelerationRatio]


def SetJOGCmd(api, isJoint, cmd, isQueued=0):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        if cmd == 9 or cmd == 10:
            tempSlaveId = -1
        else:
            tempSlaveId = slaveId
    else:
        tempSlaveId = slaveId

    cmdParam = JOGCmd()
    cmdParam.isJoint = isJoint
    cmdParam.cmd = cmd
    queuedCmdIndex = c_uint64(0)

    if cmd == 0:
        while(True):
            result = api.SetJOGCmd(c_int(masterId), c_int(-1), byref(cmdParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while(True):
            result = api.SetJOGCmd(c_int(masterId), c_int(slaveId), byref(cmdParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while(True):
            result = api.SetJOGCmd(c_int(masterId), c_int(tempSlaveId), byref(cmdParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    return [queuedCmdIndex.value]


def SetPTPJointParams(api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration, j4Velocity, j4Acceleration, isQueued=0):
    pbParam = PTPJointParams()
    pbParam.joint1Velocity = j1Velocity
    pbParam.joint1Acceleration = j1Acceleration
    pbParam.joint2Velocity = j2Velocity
    pbParam.joint2Acceleration = j2Acceleration
    pbParam.joint3Velocity = j3Velocity
    pbParam.joint3Acceleration = j3Acceleration
    pbParam.joint4Velocity = j4Velocity
    pbParam.joint4Acceleration = j4Acceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetPTPJointParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetPTPJointParams(api):
    pbParam = PTPJointParams()
    while(True):
        result = api.GetPTPJointParams(c_int(masterId), c_int(slaveId), byref(pbParam))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.joint1Velocity,pbParam.joint1Acceleration,pbParam.joint2Velocity,pbParam.joint2Acceleration,pbParam.joint3Velocity,pbParam.joint3Acceleration,pbParam.joint4Velocity,pbParam.joint4Acceleration]


def SetPTPCoordinateParams(api, xyzVelocity, xyzAcceleration, rVelocity,  rAcceleration,  isQueued=0):
    pbParam = PTPCoordinateParams()
    pbParam.xyzVelocity = xyzVelocity
    pbParam.rVelocity = rVelocity
    pbParam.xyzAcceleration = xyzAcceleration
    pbParam.rAcceleration = rAcceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetPTPCoordinateParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetPTPCoordinateParams(api):
    pbParam = PTPCoordinateParams()
    while(True):
        result = api.GetPTPCoordinateParams(c_int(masterId), c_int(slaveId), byref(pbParam))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.xyzVelocity, pbParam.rVelocity, pbParam.xyzAcceleration, pbParam.rAcceleration]
    

def SetPTPLParams(api, velocity, acceleration, isQueued=0):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId

    param = PTPLParams()
    param.velocity = velocity
    param.acceleration = acceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetPTPLParams(c_int(masterId), c_int(tempSlaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def GetPTPLParams(api):
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    param = PTPLParams()
    while(True):
        result = api.GetPTPLParams(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.velocity,  param.acceleration]
    

def SetPTPJumpParams(api, jumpHeight, zLimit, isQueued=0):
    pbParam = PTPJumpParams()
    pbParam.jumpHeight = jumpHeight
    pbParam.zLimit = zLimit
    queuedCmdIndex = c_uint64(0)
        
    while(True):
        result = api.SetPTPJumpParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetPTPJumpParams(api):
    pbParam = PTPJumpParams()
    while(True):
        result = api.GetPTPJumpParams(c_int(masterId), c_int(slaveId), byref(pbParam))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.jumpHeight, pbParam.zLimit]


def SetPTPCommonParams(api, velocityRatio, accelerationRatio, isQueued=0):
    pbParam = PTPCommonParams()
    pbParam.velocityRatio = velocityRatio
    pbParam.accelerationRatio = accelerationRatio
    queuedCmdIndex = c_uint64(0)
    
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        while(True):
            result = api.SetPTPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while(True):
            result = api.SetPTPCommonParams(c_int(masterId), c_int(-1), byref(pbParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        while(True):
            result = api.SetPTPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    else:
        while(True):
            result = api.SetPTPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break

    return [queuedCmdIndex.value]


def GetPTPCommonParams(api):
    pbParam = PTPCommonParams()
    while(True):
        result = api.GetPTPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam ))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.velocityRatio, pbParam.accelerationRatio]
    

def SetPTPCmd(api, ptpMode, x, y, z, rHead, isQueued=0):
    cmd = PTPCmd()
    cmd.ptpMode=ptpMode
    cmd.x=x
    cmd.y=y
    cmd.z=z
    cmd.rHead=rHead
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetPTPCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(2)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetPTPWithLCmd(api, ptpMode, x, y, z, rHead, l, isQueued=0):
    cmd = PTPWithLCmd()
    cmd.ptpMode=ptpMode
    cmd.x=x
    cmd.y=y
    cmd.z=z
    cmd.rHead=rHead
    cmd.l = l
    queuedCmdIndex = c_uint64(0)

    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        cmd1 = PTPCmd()
        cmd1.ptpMode = ptpMode
        cmd1.x = x
        cmd1.y = y
        cmd1.z = z
        cmd1.rHead = rHead
        queuedCmdIndex1 = c_uint64(0)
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(-1), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
        while(True):
            result = api.SetPTPCmd(c_int(masterId), c_int(slaveId), byref(cmd1), isQueued, byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    else:
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
    return [queuedCmdIndex.value]
    

def SetCPRHoldEnable(api, isEnable):
    while(True):
        result = api.SetCPRHoldEnable(c_int(masterId), c_int(slaveId), c_bool(isEnable))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetCPRHoldEnable(api):
    isEnable = c_bool(False)
    while(True):
        result = api.GetCPRHoldEnable(c_int(masterId), c_int(slaveId), byref(isEnable))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isEnable.value]
    

def SetCPParams(api, planAcc, juncitionVel, acc, realTimeTrack = 0,  isQueued=0):
    parm = CPParams()
    parm.planAcc = planAcc
    parm.juncitionVel = juncitionVel
    parm.acc = acc
    parm.realTimeTrack = realTimeTrack
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetCPParams(c_int(masterId), c_int(slaveId), byref(parm), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetCPParams(api):
    parm = CPParams()
    while(True):
        result = api.GetCPParams(c_int(masterId), c_int(slaveId), byref(parm))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [parm.planAcc, parm.juncitionVel, parm.acc, parm.realTimeTrack]


def SetCPCmd(api, cpMode, x, y, z, velocity, isQueued=0):
    cmd = CPCmd()
    cmd.cpMode = cpMode
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.velocity = velocity
    queuedCmdIndex = c_uint64(0)

    while(True):
        result = api.SetCPCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(2)
            continue
        break
    return [queuedCmdIndex.value]


def SetCP2Cmd(api, cpMode, x, y, z, isQueued=0):
    cmd = CP2Cmd()
    cmd.cpMode = cpMode
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.velocity = c_float(100)
    queuedCmdIndex = c_uint64(0)

    while(True):
        result = api.SetCP2Cmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(2)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetCPCommonParams(api, velocityRatio, accelerationRatio, isQueued=0):
    pbParam = CPCommonParams()
    pbParam.velocityRatio = velocityRatio
    pbParam.accelerationRatio = accelerationRatio
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetCPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetCPCommonParams(api):
    pbParam = CPCommonParams()
    while(True):
        result = api.GetCPCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam ))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.velocityRatio, pbParam.accelerationRatio]
    

def SetCPLECmd(api, cpMode, x, y, z, power, isQueued=0):
    cmd = CPCmd()
    cmd.cpMode = cpMode
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.velocity = power
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetCPLECmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(2)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetARCParams(api,  xyzVelocity, rVelocity, xyzAcceleration, rAcceleration,  isQueued=0):
    param = ARCParams()
    param.xyzVelocity = xyzVelocity
    param.rVelocity = rVelocity
    param.xyzAcceleration = xyzAcceleration
    param.rAcceleration = rAcceleration
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetARCParams(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]

def GetARCParams(api):
    parm = ARCParams()
    while(True):
        result = api.GetARCParams(c_int(masterId), c_int(slaveId), byref(parm))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [parm.xyzVelocity, parm.rVelocity, parm.xyzAcceleration, parm.rAcceleration]
    

def SetARCCmd(api, cirPoint, toPoint,  isQueued=0):
    cmd = ARCCmd()
    cmd.cirPoint.x = cirPoint[0];cmd.cirPoint.y = cirPoint[1];cmd.cirPoint.z = cirPoint[2];cmd.cirPoint.rHead = cirPoint[3]
    cmd.toPoint.x = toPoint[0];cmd.toPoint.y = toPoint[1];cmd.toPoint.z = toPoint[2];cmd.toPoint.rHead = toPoint[3]
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetARCCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetCircleCmd(api, cirPoint, toPoint,  isQueued=0):
    cmd = CircleCmd()
    cmd.cirPoint.x = cirPoint[0];cmd.cirPoint.y = cirPoint[1];cmd.cirPoint.z = cirPoint[2];cmd.cirPoint.rHead = cirPoint[3]
    cmd.toPoint.x = toPoint[0];cmd.toPoint.y = toPoint[1];cmd.toPoint.z = toPoint[2];cmd.toPoint.rHead = toPoint[3]
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetCircleCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetARCCommonParams(api, velocityRatio, accelerationRatio, isQueued=0):
    pbParam = ARCCommonParams()
    pbParam.velocityRatio = velocityRatio
    pbParam.accelerationRatio = accelerationRatio
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetARCCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetARCCommonParams(api):
    pbParam = ARCCommonParams()
    while(True):
        result = api.GetARCCommonParams(c_int(masterId), c_int(slaveId), byref(pbParam ))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [pbParam.velocityRatio, pbParam.accelerationRatio]


def SetWAITCmd(api, waitTime, isQueued=0):
    param = WAITCmd()
    param.waitTime = int(waitTime)
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetWAITCmd(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetTRIGCmd(api, address, mode,  condition,  threshold,  isQueued=0):
    param = TRIGCmd()
    param.address = address
    param.mode = mode
    param.condition = condition
    param.threshold = threshold
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetTRIGCmd(c_int(masterId), c_int(slaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetIOMultiplexing(api, address, multiplex, isQueued=0):
    param = IOMultiplexing()
    param.address = address
    param.multiplex = multiplex
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetIOMultiplexing(c_int(masterId), c_int(tempSlaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIOMultiplexing(api,  addr):
    param = IOMultiplexing()
    param.address = addr
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetIOMultiplexing(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.multiplex]


def SetIODO(api, address, level, isQueued=0):
    param = IODO()
    param.address = address
    param.level = level
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetIODO(c_int(masterId), c_int(tempSlaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIODO(api,  addr):
    param = IODO()
    param.address = addr
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetIODO(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.level]


def SetIOPWM(api, address, frequency, dutyCycle,  isQueued=0):
    param = IOPWM()
    param.address = address
    param.frequency = frequency
    param.dutyCycle = dutyCycle
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetIOPWM(c_int(masterId), c_int(tempSlaveId), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIOPWM(api,  addr):
    param = IOPWM()
    param.address = addr
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetIOPWM(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.frequency,  param.dutyCycle]


def GetIODI(api, addr):
    param = IODI()
    param.address = addr
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetIODI(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.level]
    

def SetEMotor(api, index, isEnabled, speed,  isQueued=0):
    emotor = EMotor()
    emotor.index = index
    emotor.isEnabled = isEnabled
    emotor.speed = speed
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetEMotor(c_int(masterId), c_int(tempSlaveId), byref(emotor), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def SetEMotorS(api, index, isEnabled, speed, distance,  isQueued=0):
    emotorS = EMotorS()
    emotorS.index = index
    emotorS.isEnabled = isEnabled
    emotorS.speed = speed
    emotorS.distance = distance
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetEMotorS(c_int(masterId), c_int(tempSlaveId), byref(emotorS), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIOADC(api, addr):
    param = IOADC()
    param.address = addr
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetIOADC(c_int(masterId), c_int(tempSlaveId), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.value]


def SetAngleSensorStaticError(api,  rearArmAngleError, frontArmAngleError):
    c_rearArmAngleError = c_float(rearArmAngleError)
    c_frontArmAngleError = c_float(frontArmAngleError)
    while(True):
        result = api.SetAngleSensorStaticError(c_int(masterId), c_int(slaveId), c_rearArmAngleError, c_frontArmAngleError)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetAngleSensorStaticError(api):
    rearArmAngleError = c_float(0)
    frontArmAngleError = c_float(0)
    while(True):
        result = api.GetAngleSensorStaticError(c_int(masterId), c_int(slaveId), byref(rearArmAngleError),  byref(frontArmAngleError))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [rearArmAngleError.value, frontArmAngleError.value]
    

def SetAngleSensorCoef(api,  rearArmAngleCoef, frontArmAngleCoef):
    c_rearArmAngleCoef = c_float(rearArmAngleCoef)
    c_frontArmAngleCoef = c_float(frontArmAngleCoef)
    while(True):
        result = api.SetAngleSensorCoef(c_int(masterId), c_int(slaveId), c_rearArmAngleCoef, c_frontArmAngleCoef)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetAngleSensorCoef(api):
    rearArmAngleCoef = c_float(0)
    frontArmAngleCoef = c_float(0)
    while(True):
        result = api.GetAngleSensorCoef(c_int(masterId), c_int(slaveId), byref(rearArmAngleCoef),  byref(frontArmAngleCoef))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [rearArmAngleCoef.value, frontArmAngleCoef.value]


def SetBaseDecoderStaticError(api,  baseDecoderError):
    c_baseDecoderError = c_float(baseDecoderError)
    while(True):
        result = api.SetBaseDecoderStaticError(c_int(masterId), c_int(slaveId), c_baseDecoderError)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def GetBaseDecoderStaticError(api):
    baseDecoderError = c_float(0)
    while(True):
        result = api.GetBaseDecoderStaticError(c_int(masterId), c_int(slaveId), byref(baseDecoderError))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [baseDecoderError.value]



def GetWIFIConnectStatus(api):
    isConnected = c_bool(0)
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIConnectStatus(c_int(masterId), c_int(slaveId), byref(isConnected))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isConnected.value]

def SetWIFIConfigMode(api,  enable):
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFIConfigMode(c_int(masterId), c_int(slaveId), enable)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def GetWIFIConfigMode(api):
    isEnabled = c_bool(0)
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIConfigMode(c_int(masterId), c_int(slaveId), byref(isEnabled))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isEnabled.value]
    

def SetWIFISSID(api,  ssid):
    szPara = create_string_buffer(len(ssid))
    szPara.raw = ssid.encode("utf-8")
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFISSID(c_int(masterId), c_int(slaveId), szPara)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def GetWIFISSID(api):
    szPara = create_string_buffer(100)
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFISSID(c_int(masterId), c_int(slaveId), szPara,  25)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    ssid = szPara.value.decode("utf-8") 
    return [ssid]
    

def SetWIFIPassword(api,  password):
    szPara = create_string_buffer(25)
    szPara.raw = password.encode("utf-8")
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFIPassword(c_int(masterId), c_int(slaveId), szPara)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetWIFIPassword(api):
    szPara = create_string_buffer(25)  
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIPassword(c_int(masterId), c_int(slaveId), szPara,  25)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    password = szPara.value.decode("utf-8") 
    return [password]
    

def SetWIFIIPAddress(api,  dhcp,  addr1,  addr2,  addr3,  addr4):
    wifiIPAddress = WIFIIPAddress()
    wifiIPAddress.dhcp = dhcp
    wifiIPAddress.addr1 = addr1
    wifiIPAddress.addr2 = addr2
    wifiIPAddress.addr3 = addr3
    wifiIPAddress.addr4 = addr4

    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFIIPAddress(c_int(masterId), c_int(slaveId), byref(wifiIPAddress))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetWIFIIPAddress(api):
    wifiIPAddress = WIFIIPAddress()
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIIPAddress(c_int(masterId), c_int(slaveId), byref(wifiIPAddress))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [c_uint8(wifiIPAddress.dhcp).value,  c_uint8(wifiIPAddress.addr1).value,  c_uint8(wifiIPAddress.addr2).value,   c_uint8(wifiIPAddress.addr3).value,  c_uint8(wifiIPAddress.addr4).value]
    

def SetWIFINetmask(api, addr1,  addr2,  addr3,  addr4):
    wifiNetmask = WIFINetmask()
    wifiNetmask.addr1 = addr1
    wifiNetmask.addr2 = addr2
    wifiNetmask.addr3 = addr3
    wifiNetmask.addr4 = addr4
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFINetmask(c_int(masterId), c_int(slaveId), byref(wifiNetmask))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
        

def GetWIFINetmask(api):
    wifiNetmask = WIFINetmask()
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFINetmask(c_int(masterId), c_int(slaveId), byref(wifiNetmask))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [c_uint8(wifiNetmask.addr1).value,  c_uint8(wifiNetmask.addr2).value,  c_uint8(wifiNetmask.addr3).value,  c_uint8(wifiNetmask.addr4).value]
    

def SetWIFIGateway(api, addr1,  addr2,  addr3,  addr4):
    wifiGateway = WIFIGateway()
    wifiGateway.addr1 = addr1
    wifiGateway.addr2 = addr2
    wifiGateway.addr3 = addr3
    wifiGateway.addr4 = addr4
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFIGateway(c_int(masterId), c_int(slaveId), byref(wifiGateway))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetWIFIGateway(api):
    wifiGateway = WIFIGateway()
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIGateway(c_int(masterId), c_int(slaveId), byref(wifiGateway))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [c_uint8(wifiGateway.addr1).value,  c_uint8(wifiGateway.addr2).value,  c_uint8(wifiGateway.addr3).value,  c_uint8(wifiGateway.addr4).value]
    

def SetWIFIDNS(api, addr1,  addr2,  addr3,  addr4):
    wifiDNS = WIFIDNS()
    wifiDNS.addr1 = addr1
    wifiDNS.addr2 = addr2
    wifiDNS.addr3 = addr3
    wifiDNS.addr4 = addr4
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.SetWIFIDNS(c_int(masterId), c_int(slaveId), byref(wifiDNS))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetWIFIDNS(api):
    wifiDNS = WIFIDNS()
    while(True):
        if not QuitDobotApiFlag:
            break
        result = api.GetWIFIDNS(c_int(masterId), c_int(slaveId), byref(wifiDNS))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [c_uint8(wifiDNS.addr1).value,  c_uint8(wifiDNS.addr2).value,  c_uint8(wifiDNS.addr3).value,  c_uint8(wifiDNS.addr4).value]


def SetColorSensor(api, isEnable, colorPort, version=0):
    enable = c_bool(isEnable)
    port = c_uint8(colorPort)
    version = c_uint8(version)
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetColorSensor(c_int(masterId), c_int(tempSlaveId), enable, port, version, 1, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def GetColorSensor(api):
    r = c_ubyte(0)
    g = c_ubyte(0)
    b = c_ubyte(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetColorSensor(c_int(masterId), c_int(tempSlaveId), byref(r),  byref(g),  byref(b))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [r.value, g.value, b.value]
    

def SetInfraredSensor(api,  isEnable, infraredPort, version=0):
    enable = c_bool(isEnable)
    port = c_uint8(infraredPort)
    queuedCmdIndex = c_uint64(0)
    version = c_uint8(version)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetInfraredSensor(c_int(masterId), c_int(tempSlaveId), enable, port, version, 1, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    

def GetInfraredSensor(api, infraredPort):
    port = c_uint8(infraredPort)
    value = c_ubyte(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetInfraredSensor(c_int(masterId), c_int(tempSlaveId), port,  byref(value))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [value.value]





def SetLostStepParams(api, threshold, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    t = c_float(threshold)
    while(True):
        result = api.SetLostStepParams(c_int(masterId), c_int(slaveId), t, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetLostStepCmd(api, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetLostStepCmd(c_int(masterId), c_int(slaveId), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]
    

def GetUART4PeripheralsType(api):
    type = c_uint8(0)
    if (masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite) or (masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle):
        while(True):
            result = api.GetUART4PeripheralsType(c_int(masterId), c_int(-1), byref(type))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break 
    elif masterDevType == DevType.Magician:
        while(True):
            result = api.GetUART4PeripheralsType(c_int(masterId), c_int(slaveId), byref(type))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
    return [type.value]
    

def GetDeviceVersionEx(api):       #2019.6.25 song 控制盒+Magician Lite时，获取控制盒的版本
    # majorVersion = c_byte(0)
    # minorVersion = c_byte(0)
    # revision     = c_byte(0)
    # hwVersion    = c_byte(0)
    deviceVersion1 = DeviceVersion()
    deviceVersion2 = DeviceVersion()
    if masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        # 2019.09.03 by song 控制盒+magicianLite 返回两个设备的版本信息
        while(True):
            result = api.GetDeviceVersion(c_int(masterId), c_int(-1), byref(deviceVersion1))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        list_MagicBoxVersion = [deviceVersion1.fw_majorVersion, deviceVersion1.fw_minorVersion, deviceVersion1.fw_revision, deviceVersion1.fw_alphaVersion,
                                deviceVersion1.hw_majorVersion, deviceVersion1.hw_minorVersion, deviceVersion1.hw_revision, deviceVersion1.hw_alphaVersion]
        while(True):
            result = api.GetDeviceVersion(c_int(masterId), c_int(slaveId), byref(deviceVersion2))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(5)
                continue
            break
        list_MagicianLiteVersion = [deviceVersion2.fw_majorVersion, deviceVersion2.fw_minorVersion, deviceVersion2.fw_revision, deviceVersion2.fw_alphaVersion,
                                    deviceVersion2.hw_majorVersion, deviceVersion2.hw_minorVersion, deviceVersion2.hw_revision, deviceVersion2.hw_alphaVersion]
        return [list_MagicBoxVersion, list_MagicianLiteVersion]

        
##################  Ex扩展函数，该套函数会检测每一条指令运行完毕  ##################
def GetPoseEx(api,  index):
    if index == 0:
        ret = GetDeviceWithL(api)
        if not ret:
            print("Dobot is not in L model")
            return
            
        lr = GetPoseL(api)
        return round(lr[0],  4)
        
    pos = GetPose(api)
    return round(pos[index-1],  4)
    
def SetHOMECmdEx(api,  temp,  isQueued=0):
    ret = SetHOMECmd(api, temp,  isQueued)
    queuedCmdIndex = c_uint64(0)
    queuedCmdIndex1 = c_uint64(0)
    if masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        if isUsingLinearRail:        
            while(True):
                result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
                if result == DobotCommunicate.DobotCommunicate_NoError and ret[1] <= queuedCmdIndex1.value:
                    break
                dSleep(100)
            while(True):
                result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex))
                if result == DobotCommunicate.DobotCommunicate_NoError and ret[0] <= queuedCmdIndex.value:
                    break
                dSleep(100)
        else:
            while(True):
                result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex))
                if result == DobotCommunicate.DobotCommunicate_NoError and ret[0] <= queuedCmdIndex.value:
                    break
                dSleep(100)
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.Idle: 
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
            if result == DobotCommunicate.DobotCommunicate_NoError and ret[1] <= queuedCmdIndex1.value:
                break
            dSleep(100)
    else:
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex))
            if result == DobotCommunicate.DobotCommunicate_NoError and ret[0] <= queuedCmdIndex.value:
                break
            dSleep(100)
        
def SetWAITCmdEx(api, waitTime, isQueued=0):
    ret = SetWAITCmd(api, waitTime, isQueued)
    while(True):
        if not QuitDobotApiFlag:
            break
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
           break
    # dSleep(waitTime * 1000)
    
def SetEndEffectorParamsEx(api, xBias, yBias, zBias, isQueued=0):
    ret = SetEndEffectorParams(api, xBias, yBias, zBias, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
        
def SetPTPJointParamsEx(api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration, j4Velocity, j4Acceleration, isQueued=0):
    ret = SetPTPJointParams(api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration, j4Velocity, j4Acceleration, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
        
def SetPTPCoordinateParamsEx(api, xyzVelocity, xyzAcceleration, rVelocity,  rAcceleration,  isQueued=0):
    ret = SetPTPCoordinateParams(api, xyzVelocity, xyzAcceleration, rVelocity,  rAcceleration,  isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)

def SetPTPLParamsEx(api, lVelocity, lAcceleration, isQueued=0):
    ret = GetDeviceWithL(api)
    if not ret:
        print("Dobot is not in L model")
        return
    
    ret = SetPTPLParams(api, lVelocity, lAcceleration, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
        
def SetPTPCommonParamsEx(api, velocityRatio, accelerationRatio, isQueued=0):
    ret = SetPTPCommonParams(api, velocityRatio, accelerationRatio, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
        
def SetPTPJumpParamsEx(api, jumpHeight, maxJumpHeight, isQueued=0):
    ret = SetPTPJumpParams(api, jumpHeight, maxJumpHeight, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
        
def SetPTPCmdEx(api, ptpMode, x, y, z, rHead, isQueued=0):
    ret = SetPTPCmd(api, ptpMode, x, y, z, rHead, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)
    
def SetIOMultiplexingEx(api, address, multiplex, isQueued=0):
    ret = SetIOMultiplexing(api, address, multiplex, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)
        
def SetEndEffectorSuctionCupEx(api, enableCtrl,  on, isQueued=0):
    ret = SetEndEffectorSuctionCup(api, enableCtrl,  on, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)

def SetEndEffectorGripperEx(api, enableCtrl,  on, isQueued=0):
    ret = SetEndEffectorGripper(api, enableCtrl,  on, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
        
def SetEndEffectorLaserEx(api, enableCtrl, power, isQueued=0):
    SetIOMultiplexingEx(api, 2,  1, isQueued)
    SetIOMultiplexingEx(api, 4,  2, isQueued)
    SetIODOEx(api, 2, enableCtrl, isQueued)
    SetIOPWMEx(api, 4, 10000, power, isQueued)

def SetIODOEx(api, address, level, isQueued=0):
    ret = SetIODO(api, address, level, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)
        
def SetEMotorEx(api, index, isEnabled, speed,  isQueued=0):
    ret = SetEMotor(api, index, isEnabled, speed,  isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)
    
def SetEMotorSEx(api, index, isEnabled, speed, distance,  isQueued=0):
    ret = SetEMotorS(api, index, isEnabled, speed, distance,   isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)
    
def SetIOPWMEx(api, address, frequency, dutyCycle,  isQueued=0):
    ret = SetIOPWM(api, address, frequency, dutyCycle,  isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetPTPWithLCmdEx(api, ptpMode, x, y, z, rHead,  l, isQueued=0):
    ret = GetDeviceWithL(api)
    if not ret:
        print("Dobot is not in L model")
        return

    cmd = PTPWithLCmd()
    cmd.ptpMode=ptpMode
    cmd.x=x
    cmd.y=y
    cmd.z=z
    cmd.rHead=rHead
    cmd.l = l
    queuedCmdIndex = c_uint64(0)
    queuedCmdIndex1 = c_uint64(0)
    queuedCmdIndex2 = c_uint64(0)
    # 滑轨的特殊处理
    if slaveDevType == DevType.Magician:
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError or queuedCmdIndex1.value < queuedCmdIndex.value:
                dSleep(2)
                continue
            break
    elif masterDevType == DevType.Conntroller and slaveDevType == DevType.MagicianLite:
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(-1), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            queuedCmdIndex2 = queuedCmdIndex
            break
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError or queuedCmdIndex1.value < queuedCmdIndex2.value:
                dSleep(2)
                continue
            break

        while(True):
            result = api.SetPTPCmd(c_int(masterId), c_int(slaveId), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            break
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(slaveId), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError or queuedCmdIndex1.value < queuedCmdIndex.value:
                dSleep(2)
                continue
            break
    else:
        while(True):
            result = api.SetPTPWithLCmd(c_int(masterId), c_int(-1), byref(cmd), isQueued, byref(queuedCmdIndex))
            if result != DobotCommunicate.DobotCommunicate_NoError:
                dSleep(2)
                continue
            queuedCmdIndex2 = queuedCmdIndex
            break
        while(True):
            result = api.GetQueuedCmdCurrentIndex(c_int(masterId), c_int(-1), byref(queuedCmdIndex1))
            if result != DobotCommunicate.DobotCommunicate_NoError or queuedCmdIndex1.value < queuedCmdIndex.value:
                dSleep(2)
                continue
            break
    return [queuedCmdIndex2.value]


def GetColorSensorEx(api,  index):
    result = GetColorSensor(api)
    return result[index]

    
def SetAutoLevelingCmdEx(api, controlFlag, precision, isQueued=1):
    index = SetAutoLevelingCmd(api, controlFlag, precision, isQueued)[0]
    while(True):
        if index <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)

   
def SetLostStepCmdEx(api, isQueued=1):
    ret = SetLostStepCmd(api, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)


def SetUpgradeFWReadyCmd(api,fwSize, md5):
    upgradeFWReadyCmd = UpgradeFWReadyCmd()
    upgradeFWReadyCmd.fwSize = fwSize
    try:
        md5Bytes = bytes.fromhex(md5)
        md5CBuf = create_string_buffer(len(md5Bytes))
        md5CBuf.raw = md5Bytes
        upgradeFWReadyCmd.md5 = addressof(md5CBuf)
    except Exception as e:
        print(e)

    # # 只发送给主设备
    # result = api.SetUpgradeFWReadyCmd(c_int(masterId), c_int(-1), byref(upgradeFWReadyCmd))
    # return result

    # 不能去掉等待！！！！！！，jomar 2019年5月7日 09:28:30
    if masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetUpgradeFWReadyCmd(c_int(masterId), c_int(tempSlaveId), byref(upgradeFWReadyCmd))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def GetUpgradeFWReadyCmd(api,fwSize, md5):
    upgradeFWReadyCmd = UpgradeFWReadyCmd()
    upgradeFWReadyCmd.fwSize = fwSize
    isUpgrade = c_byte(0)
    try:
        md5Bytes = bytes.fromhex(md5)
        md5CBuf = create_string_buffer(len(md5Bytes))
        md5CBuf.raw = md5Bytes
        upgradeFWReadyCmd.md5 = addressof(md5CBuf)
    except Exception as e:
        print(e)

    # # 只发送给主设备
    # result = api.SetUpgradeFWReadyCmd(c_int(masterId), c_int(-1), byref(upgradeFWReadyCmd))
    # return result

    # 不能去掉等待！！！！！！，jomar 2019年5月7日 09:28:30
    if masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetUpgradeFWReadyCmd(c_int(masterId), c_int(tempSlaveId), byref(upgradeFWReadyCmd), byref(isUpgrade))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [isUpgrade.value]






# jomar, 2019年5月9日 10:10:50


def SetTRIGCmdEx(api, address, mode,  condition,  threshold,  isQueued=1):
    ret = SetTRIGCmd(api, address, mode, condition, threshold, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)


def SetARCCmdEx(api, cirPoint, toPoint, isQueued=1):
    ret = SetARCCmd(api, cirPoint, toPoint, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)


def SetMotorMode(api, mode):
    while(True):
        result = api.SetMotorMode(c_int(masterId), c_int(slaveId), c_int(mode))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break 


def GetMotorMode(api):
    mode = c_int(0)
    while(True):
        result = api.GetMotorMode(c_int(masterId), c_int(slaveId), byref(mode))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break 
    return [mode.value]



#BLOCKLY 2019-04-29 控制盒IO

def SetIOMultiplexingExt(api, address, multiplex, isQueued=0):
    param = IOMultiplexing()
    param.address = address
    param.multiplex = multiplex
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetIOMultiplexing(c_int(masterId), c_int(-1), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIOMultiplexingExt(api, addr):
    param = IOMultiplexing()
    param.address = addr
    while(True):
        result = api.GetIOMultiplexing(c_int(masterId), c_int(-1), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.multiplex]


def GetIOADCExt(api, addr):
    param = IOADC()
    param.address = addr
    while(True):
        result = api.GetIOADC(c_int(masterId), c_int(-1), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.value]


def SetIOPWMExt(api, address, frequency, dutyCycle,  isQueued=0):
    param = IOPWM()
    param.address = address
    param.frequency = frequency
    param.dutyCycle = dutyCycle
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetIOPWM(c_int(masterId), c_int(-1), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIOPWMExt(api, addr):
    param = IOPWM()
    param.address = addr
    while(True):
        result = api.GetIOPWM(c_int(masterId), c_int(-1), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.frequency,  param.dutyCycle]


def GetIODIExt(api, addr):
    param = IODI()
    param.address = addr
    while(True):
        result = api.GetIODI(c_int(masterId), c_int(-1), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.level]


def SetIODOExt(api, address, level, isQueued=0):
    param = IODO()
    param.address = address
    param.level = level
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetIODO(c_int(masterId), c_int(-1), byref(param), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetIODOExt(api, addr):
    param = IODO()
    param.address = addr
    while(True):
        result = api.GetIODO(c_int(masterId), c_int(-1), byref(param))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [param.level]


def SetEMotorExt(api, index, isEnabled, speed, isQueued=0):
    emotor = EMotor()
    emotor.index = index
    emotor.isEnabled = isEnabled
    emotor.speed = speed
    queuedCmdIndex = c_uint64(0)
    while (True):
        result = api.SetEMotor(c_int(masterId), c_int(-1), byref(emotor), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetEMotorSExt(api, index, isEnabled, speed, distance, isQueued=0):
    emotorS = EMotorS()
    emotorS.index = index
    emotorS.isEnabled = isEnabled
    emotorS.speed = speed
    emotorS.distance = distance
    queuedCmdIndex = c_uint64(0)
    while (True):
        result = api.SetEMotorS(c_int(masterId), c_int(-1), byref(emotorS), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetColorSensorExt(api, isEnable, colorPort, version=0, isQueued=0):
    enable = c_bool(isEnable)
    port = c_uint8(colorPort)
    version = c_uint8(version)
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetColorSensor(c_int(masterId), c_int(-1), enable, port, version, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def SetInfraredSensorExt(api,  isEnable, infraredPort, version=0, isQueued=0):
    enable = c_bool(isEnable)
    port = c_uint8(infraredPort)
    version = c_uint8(version)
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetInfraredSensor(c_int(masterId), c_int(-1), enable, port, version, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetInfraredSensorExt(api, infraredPort):
    port = c_uint8(infraredPort)
    value = c_ubyte(0)
    
    while(True):
        result = api.GetInfraredSensor(c_int(masterId), c_int(-1), port,  byref(value))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [value.value]


def GetColorSensorExt(api, index):
    r = c_ubyte(0)
    g = c_ubyte(0)
    b = c_ubyte(0)
    while(True):
        result = api.GetColorSensor(c_int(masterId), c_int(-1), byref(r),  byref(g),  byref(b))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [r.value, g.value, b.value][index]

# 控制盒IO同步

def SetIOMultiplexingExtEx(api, address, multiplex, isQueued=0):
    ret = SetIOMultiplexingExt(api, address, multiplex, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)

def SetIOPWMExtEx(api, address, frequency, dutyCycle,  isQueued=0):
    ret = SetIOPWMExt(api, address, frequency, dutyCycle,  isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetIODOExtEx(api, address, level, isQueued=0):
    ret = SetIODOExt(api, address, level, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetEMotorExtEx(api, index, isEnabled, speed, isQueued=0):
    ret = SetEMotorExt(api, index, isEnabled, speed, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetEMotorSExtEx(api, index, isEnabled, speed, distance, isQueued=0):
    ret = SetEMotorSExt(api, index, isEnabled, speed, distance, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetColorSensorExtEx(api, isEnable, colorPort, version=0, isQueued=0):
    ret = SetColorSensorExt(api, isEnable, colorPort, version, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetInfraredSensorExtEx(api,  isEnable, infraredPort, version=0, isQueued=0):
    ret = SetInfraredSensorExt(api,  isEnable, infraredPort, version, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


#2019.08.21 by song add Seeed Sensor API    

def GetSeeedColorSensorExt(api):
    r = c_ushort(0)
    g = c_ushort(0)
    b = c_ushort(0)
    Cct = c_ushort(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetSeeedColorSensor(c_int(masterId), c_int(tempSlaveId), byref(r),  byref(g),  byref(b), byref(Cct))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [r.value, g.value, b.value, Cct.value]


def SetSeeedColorSensorExt(api, SeeedPort,isQueued=0):
    queuedCmdIndex = c_uint64(0)
    port = c_uint8(SeeedPort)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetSeeedColorSensor(c_int(masterId), c_int(tempSlaveId), port, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetSeeedDistanceSensorExt(api, SeeedPort):
    port = c_uint8(SeeedPort)
    distance = c_ubyte(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetSeeedDistanceSensor(c_int(masterId), c_int(tempSlaveId), port, byref(distance))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [distance.value]


def SetSeeedTempSensorExt(api, SeeedPort, isQueued=0):
    port = c_uint8(SeeedPort)
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetSeeedTempSensor(c_int(masterId), c_int(tempSlaveId), port, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetSeeedTempSensorExt(api):
    tem = c_ushort(0)
    hum = c_ushort(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetSeeedTempSensor(c_int(masterId), c_int(tempSlaveId), byref(tem),  byref(hum))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [tem.value, hum.value]


def SetSeeedLightSensorExt(api, SeeedPort, isQueued=0):
    port = c_uint8(SeeedPort)
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetSeeedLightSensor(c_int(masterId), c_int(tempSlaveId), port, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetSeeedLightSensorExt(api):
    lux = c_ushort(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.GetSeeedLightSensor(c_int(masterId), c_int(tempSlaveId), byref(lux))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [lux.value]


def SetSeeedRgbExt(api, SeeedPort, Rgb, isQueued=0):
    port = c_ubyte(SeeedPort)
    rgb = c_float(Rgb)
    queuedCmdIndex = c_uint64(0)
    if slaveDevType == DevType.Magician:
        tempSlaveId = slaveId
    elif masterDevType == DevType.Conntroller and (slaveDevType == DevType.MagicianLite or slaveDevType == DevType.Idle):
        tempSlaveId = -1
    else:
        tempSlaveId = slaveId
    while(True):
        result = api.SetSeeedRgb(c_int(masterId), c_int(tempSlaveId), port, rgb, isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]

# seeed传感器同步指令

def SetSeeedColorSensorExtEx(api, SeeedPort,isQueued=0):
    ret = SetSeeedColorSensorExt(api, SeeedPort, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetSeeedTempSensorExtEx(api, SeeedPort, isQueued=0):
    ret = SetSeeedTempSensorExt(api, SeeedPort, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetSeeedLightSensorExtEx(api, SeeedPort, isQueued=0):
    ret = SetSeeedLightSensorExt(api, SeeedPort, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)


def SetSeeedRgbExtEx(api, SeeedPort, Rgb, isQueued=0):
    ret = SetSeeedRgbExt(api, SeeedPort, Rgb, isQueued)
    if masterDevType == DevType.Magician:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
                break
            dSleep(5)
    else:
        while(True):
            if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
                break
            dSleep(5)
    

def RestartMagicBox(api):
    while(True):
        result = api.RestartMagicBox(c_int(masterId), c_int(-1))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


#Magician Lite 2019-11-05 Magician Lite单独的API


def SetLostStepEnableAndParamsCmd(api, enable, threshlod, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetLostStepEnableAndParamsCmd(c_int(masterId), c_int(slaveId), c_uint8(enable), c_float(threshlod), isQueued, byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetLostStepEnableAndParamsCmd(api):
    enable = c_uint8(0)
    threshlod = c_float(0)
    while(True):
        result = api.GetLostStepEnableAndParamsCmd(c_int(masterId), c_int(slaveId), byref(enable), byref(threshlod))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [enable.value, threshlod.value]



def SetEndEffectorType(api, endType=0, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetEndEffectorType(c_int(masterId), c_int(slaveId), isQueued, c_uint8(endType), byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break  
    return[queuedCmdIndex.value]


def GetEndEffectorType(api):
    endType = c_uint8(0)
    while(True):
        result = api.GetEndEffectorType(c_int(masterId), c_int(slaveId), byref(endType))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [endType.value]


def SetServoAngle(api, servoId, angle, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetServoAngle(c_int(masterId), c_int(-1), isQueued, c_uint8(servoId), c_float(angle), byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break 
    return [queuedCmdIndex.value]


def GetServoAngle(api, servoId):
    angle = c_float(0)
    while(True):
        result = api.GetServoAngle(c_int(masterId), c_int(-1),  c_uint8(servoId) ,byref(angle))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [angle.value]


def SetArmSpeedRatio(api, paramsMode, speedRatio, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetArmSpeedRatio(c_int(masterId), c_int(slaveId), isQueued, c_uint8(paramsMode), c_uint8(speedRatio),  byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break 
    return [queuedCmdIndex.value]


def GetArmSpeedRatio(api, paramsMode=0):
    speedRatio = c_uint8(0)
    # paramsMode = c_uint8(0)
    while(True):
        result = api.GetArmSpeedRatio(c_int(masterId), c_int(slaveId),  c_uint8(paramsMode), byref(speedRatio))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return[speedRatio.value]


def SetLSpeedRatio(api, paramsMode, speedRatio, isQueued=0):
    queuedCmdIndex = c_uint64(0)
    while(True):
        result = api.SetLSpeedRatio(c_int(masterId), c_int(-1), isQueued, c_uint8(paramsMode), c_uint8(speedRatio), byref(queuedCmdIndex))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return [queuedCmdIndex.value]


def GetLSpeedRatio(api, paramsMode):
    speedRatio = c_uint8(0)
    while(True):
        result = api.GetLSpeedRatio(c_int(masterId), c_int(-1), c_uint8(paramsMode), byref(speedRatio))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break
    return[speedRatio.value]


def PrintInfo(api, info):
    szPara = create_string_buffer(len(info))
    szPara.raw = info.encode("utf-8")
    while(True):
        result = api.PrintInfo(c_int(masterId), c_int(-1), szPara)
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break


def SetProgbar(api, progbar):
    while(True):
        result = api.SetProgbar(c_int(masterId), c_int(-1), c_uint8(progbar))
        if result != DobotCommunicate.DobotCommunicate_NoError:
            dSleep(5)
            continue
        break

#MagicianLite/Magic Box同步等待

def SetEndEffectorTypeEx(api, endType=0, isQueued=1):
    ret = SetEndEffectorType(api, endType, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)


def SetServoAngleEx(api, servoId, angle, isQueued=1):
    ret = SetServoAngle(api, servoId, angle, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
            break
        dSleep(5)


def SetArmSpeedRatioEx(api, paramsMode=0, speedRatio=0, isQueued=1):
    ret = SetArmSpeedRatio(api,paramsMode, speedRatio, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[0]:
            break
        dSleep(5)


def SetLSpeedRatioEx(api, paramsMode, speedRatio, isQueued=1):
    ret = SetLSpeedRatio(api, paramsMode, speedRatio, isQueued)
    while(True):
        if ret[0] <= GetQueuedCmdCurrentIndex(api)[1]:
            break
        dSleep(5)
