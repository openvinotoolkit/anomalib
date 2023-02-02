#ifndef DOBOTDLL_H
#define DOBOTDLL_H

#include "dobotdll_global.h"
#include "DobotType.h"

extern "C" DOBOTDLLSHARED_EXPORT int DobotExec(void);

extern "C" DOBOTDLLSHARED_EXPORT int SearchDobot(char *dobotNameList, uint32_t maxLen);
extern "C" DOBOTDLLSHARED_EXPORT int ConnectDobot(const char *portName, uint32_t baudrate, char *fwType, char *version);
extern "C" DOBOTDLLSHARED_EXPORT int DisconnectDobot(void);

extern "C" DOBOTDLLSHARED_EXPORT int SetCmdTimeout(uint32_t cmdTimeout);

// Device information
extern "C" DOBOTDLLSHARED_EXPORT int SetDeviceSN(const char *deviceSN);
extern "C" DOBOTDLLSHARED_EXPORT int GetDeviceSN(char *deviceSN, uint32_t maxLen);

extern "C" DOBOTDLLSHARED_EXPORT int SetDeviceName(const char *deviceName);
extern "C" DOBOTDLLSHARED_EXPORT int GetDeviceName(char *deviceName, uint32_t maxLen);

extern "C" DOBOTDLLSHARED_EXPORT int GetDeviceVersion(uint8_t *majorVersion, uint8_t *minorVersion, uint8_t *revision);

extern "C" DOBOTDLLSHARED_EXPORT int SetDeviceWithL(bool isWithL, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetDeviceWithL(bool *isWithL);

extern "C" DOBOTDLLSHARED_EXPORT int GetDeviceTime(uint32_t *deviceTime);

// Pose and Kinematics parameters are automatically get
extern "C" DOBOTDLLSHARED_EXPORT int GetPose(Pose *pose);
extern "C" DOBOTDLLSHARED_EXPORT int ResetPose(bool manual, float rearArmAngle, float frontArmAngle);
extern "C" DOBOTDLLSHARED_EXPORT int GetKinematics(Kinematics *kinematics);
extern "C" DOBOTDLLSHARED_EXPORT int GetPoseL(float *l);

// Alarms
extern "C" DOBOTDLLSHARED_EXPORT int GetAlarmsState(uint8_t *alarmsState, uint32_t *len, uint32_t maxLen);
extern "C" DOBOTDLLSHARED_EXPORT int ClearAllAlarmsState(void);

// HOME
extern "C" DOBOTDLLSHARED_EXPORT int SetHOMEParams(HOMEParams *homeParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetHOMEParams(HOMEParams *homeParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetHOMECmd(HOMECmd *homeCmd, bool isQueued, uint64_t *queuedCmdIndex);

extern "C" DOBOTDLLSHARED_EXPORT int SetAutoLevelingCmd(AutoLevelingCmd *autoLevelingCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetAutoLevelingResult(float *precision);

// Handheld teach
extern "C" DOBOTDLLSHARED_EXPORT int SetHHTTrigMode(HHTTrigMode hhtTrigMode);
extern "C" DOBOTDLLSHARED_EXPORT int GetHHTTrigMode(HHTTrigMode *hhtTrigMode);

extern "C" DOBOTDLLSHARED_EXPORT int SetHHTTrigOutputEnabled(bool isEnabled);
extern "C" DOBOTDLLSHARED_EXPORT int GetHHTTrigOutputEnabled(bool *isEnabled);

extern "C" DOBOTDLLSHARED_EXPORT int GetHHTTrigOutput(bool *isTriggered);

// EndEffector
extern "C" DOBOTDLLSHARED_EXPORT int SetEndEffectorParams(EndEffectorParams *endEffectorParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetEndEffectorParams(EndEffectorParams *endEffectorParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetEndEffectorLaser(bool enableCtrl, bool on, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetEndEffectorLaser(bool *isCtrlEnabled, bool *isOn);

extern "C" DOBOTDLLSHARED_EXPORT int SetEndEffectorSuctionCup(bool enableCtrl, bool suck, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetEndEffectorSuctionCup(bool *isCtrlEnabled, bool *isSucked);

extern "C" DOBOTDLLSHARED_EXPORT int SetEndEffectorGripper(bool enableCtrl, bool grip, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetEndEffectorGripper(bool *isCtrlEnabled, bool *isGripped);

// Arm orientation
extern "C" DOBOTDLLSHARED_EXPORT int SetArmOrientation(ArmOrientation armOrientation, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetArmOrientation(ArmOrientation *armOrientation);

// JOG functions
extern "C" DOBOTDLLSHARED_EXPORT int SetJOGJointParams(JOGJointParams *jointJogParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetJOGJointParams(JOGJointParams *jointJogParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetJOGCoordinateParams(JOGCoordinateParams *coordinateJogParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetJOGCoordinateParams(JOGCoordinateParams *coordinateJogParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetJOGLParams(JOGLParams *jogLParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetJOGLParams(JOGLParams *jogLParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetJOGCommonParams(JOGCommonParams *jogCommonParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetJOGCommonParams(JOGCommonParams *jogCommonParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetJOGCmd(JOGCmd *jogCmd, bool isQueued, uint64_t *queuedCmdIndex);

// PTP functions
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPJointParams(PTPJointParams *ptpJointParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPJointParams(PTPJointParams *ptpJointParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPCoordinateParams(PTPCoordinateParams *ptpCoordinateParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPCoordinateParams(PTPCoordinateParams *ptpCoordinateParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPLParams(PTPLParams *ptpLParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPLParams(PTPLParams *ptpLParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetPTPJumpParams(PTPJumpParams *ptpJumpParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPJumpParams(PTPJumpParams *ptpJumpParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPCommonParams(PTPCommonParams *ptpCommonParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPCommonParams(PTPCommonParams *ptpCommonParams);

extern "C" DOBOTDLLSHARED_EXPORT int SetPTPCmd(PTPCmd *ptpCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPWithLCmd(PTPWithLCmd *ptpWithLCmd, bool isQueued, uint64_t *queuedCmdIndex);

extern "C" DOBOTDLLSHARED_EXPORT int SetPTPJump2Params(PTPJump2Params *ptpJump2Params, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPJump2Params(PTPJump2Params *ptpJump2Params);

extern "C" DOBOTDLLSHARED_EXPORT int SetPTPPOCmd(PTPCmd *ptpCmd, ParallelOutputCmd *parallelCmd, int parallelCmdCount, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetPTPPOWithLCmd(PTPWithLCmd *ptpWithLCmd, ParallelOutputCmd *parallelCmd, int parallelCmdCount, bool isQueued, uint64_t *queuedCmdIndex);

// CP functions
extern "C" DOBOTDLLSHARED_EXPORT int SetCPParams(CPParams *cpParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetCPParams(CPParams *cpParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetCPCmd(CPCmd *cpCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetCPLECmd(CPCmd *cpCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetCPRHoldEnable(bool isEnable);
extern "C" DOBOTDLLSHARED_EXPORT int GetCPRHoldEnable(bool *isEnable);
extern "C" DOBOTDLLSHARED_EXPORT int SetCPCommonParams(CPCommonParams *cpCommonParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetCPCommonParams(CPCommonParams *cpCommonParams);

// ARC
extern "C" DOBOTDLLSHARED_EXPORT int SetARCParams(ARCParams *arcParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetARCParams(ARCParams *arcParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetARCCmd(ARCCmd *arcCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetCircleCmd(CircleCmd *circleCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetARCCommonParams(ARCCommonParams *arcCommonParams, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetARCCommonParams(ARCCommonParams *arcCommonParams);

// WAIT
extern "C" DOBOTDLLSHARED_EXPORT int SetWAITCmd(WAITCmd *waitCmd, bool isQueued, uint64_t *queuedCmdIndex);

// TRIG
extern "C" DOBOTDLLSHARED_EXPORT int SetTRIGCmd(TRIGCmd *trigCmd, bool isQueued, uint64_t *queuedCmdIndex);

// EIO
extern "C" DOBOTDLLSHARED_EXPORT int SetIOMultiplexing(IOMultiplexing *ioMultiplexing, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetIOMultiplexing(IOMultiplexing *ioMultiplexing);

extern "C" DOBOTDLLSHARED_EXPORT int SetIODO(IODO *ioDO, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetIODO(IODO *ioDO);

extern "C" DOBOTDLLSHARED_EXPORT int SetIOPWM(IOPWM *ioPWM, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetIOPWM(IOPWM *ioPWM);

extern "C" DOBOTDLLSHARED_EXPORT int GetIODI(IODI *ioDI);
extern "C" DOBOTDLLSHARED_EXPORT int GetIOADC(IOADC *ioADC);

extern "C" DOBOTDLLSHARED_EXPORT int SetEMotor(EMotor *eMotor, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetEMotorS(EMotorS *eMotorS, bool isQueued, uint64_t *queuedCmdIndex);

extern "C" DOBOTDLLSHARED_EXPORT int SetColorSensor(bool enable,ColorPort colorPort);
extern "C" DOBOTDLLSHARED_EXPORT int GetColorSensor(uint8_t *r, uint8_t *g, uint8_t *b);

extern "C" DOBOTDLLSHARED_EXPORT int SetInfraredSensor(bool enable,InfraredPort infraredPort);
extern "C" DOBOTDLLSHARED_EXPORT int GetInfraredSensor(InfraredPort port, uint8_t *value);

// CAL
extern "C" DOBOTDLLSHARED_EXPORT int SetAngleSensorStaticError(float rearArmAngleError, float frontArmAngleError);
extern "C" DOBOTDLLSHARED_EXPORT int GetAngleSensorStaticError(float *rearArmAngleError, float *frontArmAngleError);
extern "C" DOBOTDLLSHARED_EXPORT int SetAngleSensorCoef(float rearArmAngleCoef, float frontArmAngleCoef);
extern "C" DOBOTDLLSHARED_EXPORT int GetAngleSensorCoef(float *rearArmAngleCoef, float *frontArmAngleCoef);

extern "C" DOBOTDLLSHARED_EXPORT int SetBaseDecoderStaticError(float baseDecoderError);
extern "C" DOBOTDLLSHARED_EXPORT int GetBaseDecoderStaticError(float *baseDecoderError);

extern "C" DOBOTDLLSHARED_EXPORT int SetLRHandCalibrateValue(float lrHandCalibrateValue);
extern "C" DOBOTDLLSHARED_EXPORT int GetLRHandCalibrateValue(float *lrHandCalibrateValue);

// WIFI
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFIConfigMode(bool enable);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIConfigMode(bool *isEnabled);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFISSID(const char *ssid);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFISSID(char *ssid, uint32_t maxLen);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFIPassword(const char *password);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIPassword(char *password, uint32_t maxLen);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFIIPAddress(WIFIIPAddress *wifiIPAddress);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIIPAddress(WIFIIPAddress *wifiIPAddress);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFINetmask(WIFINetmask *wifiNetmask);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFINetmask(WIFINetmask *wifiNetmask);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFIGateway(WIFIGateway *wifiGateway);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIGateway(WIFIGateway *wifiGateway);
extern "C" DOBOTDLLSHARED_EXPORT int SetWIFIDNS(WIFIDNS *wifiDNS);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIDNS(WIFIDNS *wifiDNS);
extern "C" DOBOTDLLSHARED_EXPORT int GetWIFIConnectStatus(bool *isConnected);

//FIRMWARE
extern "C" DOBOTDLLSHARED_EXPORT int UpdateFirmware(FirmwareParams *firmwareParams);
extern "C" DOBOTDLLSHARED_EXPORT int SetFirmwareMode(FirmwareMode *firmwareMode);
extern "C" DOBOTDLLSHARED_EXPORT int GetFirmwareMode(FirmwareMode *firmwareMode);

//LOSTSTEP
extern "C" DOBOTDLLSHARED_EXPORT int SetLostStepParams(float threshold, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SetLostStepCmd(bool isQueued, uint64_t *queuedCmdIndex);

//UART4 Peripherals
extern "C" DOBOTDLLSHARED_EXPORT int GetUART4PeripheralsType(uint8_t *type);
extern "C" DOBOTDLLSHARED_EXPORT int SetUART4PeripheralsEnable(bool isEnable);
extern "C" DOBOTDLLSHARED_EXPORT int GetUART4PeripheralsEnable(bool *isEnable);

//Function Pluse Mode
extern "C" DOBOTDLLSHARED_EXPORT int SendPluse(PluseCmd *pluseCmd, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SendPluseEx(PluseCmd *pluseCmd);

// TEST
extern "C" DOBOTDLLSHARED_EXPORT int GetUserParams(UserParams *userParams);
extern "C" DOBOTDLLSHARED_EXPORT int GetPTPTime(PTPCmd *ptpCmd, uint32_t *ptpTime);
extern "C" DOBOTDLLSHARED_EXPORT int GetServoPIDParams(PID *pid);
extern "C" DOBOTDLLSHARED_EXPORT int SetServoPIDParams(PID *pid, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int GetServoControlLoop(uint8_t index, uint8_t *controlLoop);
extern "C" DOBOTDLLSHARED_EXPORT int SetServoControlLoop(uint8_t index, uint8_t controlLoop, bool isQueued, uint64_t *queuedCmdIndex);
extern "C" DOBOTDLLSHARED_EXPORT int SaveServoPIDParams(uint8_t index, uint8_t controlLoop, bool isQueued, uint64_t *queuedCmdIndex);

// Queued command
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdStartExec(void);
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdStopExec(void);
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdForceStopExec(void);
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdStartDownload(uint32_t totalLoop, uint32_t linePerLoop);
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdStopDownload(void);
extern "C" DOBOTDLLSHARED_EXPORT int SetQueuedCmdClear(void);
extern "C" DOBOTDLLSHARED_EXPORT int GetQueuedCmdCurrentIndex(uint64_t *queuedCmdCurrentIndex);

#endif // DOBOTDLL_H
