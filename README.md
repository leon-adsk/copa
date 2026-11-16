# Dataset
Three files where each one is a separate archive, two sample every 3 sec one on variable change.
Columns: Date, Time, Variable name, value, Status

The meaning of values for each variable type are described in the next section.
The meaning of each status value can be found here  [Zenon help](https://onlinehelp.copadata.com/Content.html?product=supervisor&version=1600&lang=en&id=24139), we need only ignore each row with INVALID status, the rest is not important for us.

Anomalies: between October 29 and November 4th ~10:50; after that, so Nov 4 ~10:50 - 15:00, no anomalies occurred anymore.

## Variable Descriptions
1. M07.WS_Good_packages: Count of '"good" filled bottles (this excludes rejects / scrap / defective bottles) for current order
2. M07.WS_Tot_Packages: Count of total filled bottles for current order
3. M07.WS_Cur_Mach_Spd:  Current machine speed
4. Sim/Global/M7_FillWeight: Fill Weight of the current bottle
5. Sim/Global/M7.Admin.ProductData_DefectiveCount: Count of excludes rejects / scrap / defective bottles for current order
6. Converter/Global/M07.Power: Current needed power input of the filler
7. Sim/Global/M7_Energy_incremental: Accumulated power consumption of the filler for current
8. Sim/Global/M7_CompAir_incremental: Accumulated compressed air consumption of the filler for current
9. Sim/Global/M7_CompAir: Current needed compressed air input of the filler
10. Sim/Global/M7_Power: same as 6.)
11. Sim/Global/M7.Status.CurMachSpeed: same as 3.) 
12. Sim/Global/M7.Status.MachSpeed: Target or nominal speed
13. Sim/Global/M7.Status.StateCurrent: Current state of the machine
    - 0     Comm. Error 
    - 1     Stopped
    - 2     Starting
    - 4     Prepared
    - 8     Lack
    - 16    Tailback
    - 32    Lack Branch Line
    - 64    Tailback Branch Line
    - 128   Operating
    - 256   Stopping
    - 512   Aborting
    - 1024  Equipment Failure
    - 2048  External Failure
    - 4096  Emergency Stop
    - 8192  Holding
    - 16384 Held
    - 32768 Idle
    - 65536 Unholding
    - 131072 Suspending
    - 262144 Unsuspending
    - 524288 Resetting
    - 1048576 Clearing
14. Sim/Global/M7.Status.UnitModeCurrent: 1 = OFF; 2 = Manual; 4 = Semi-Automatic; 8 = Automatic
15. Sim/Global/M7.Admin.ProductData_ProcessedCount: Count of total filled bottles for current order
16. M07.WS_Cur_State: same as 13.)
17. M07.WS_Cur_Prog: 1 = Production; 2 = Start Up; 4 = Run Down; 8 = Cleaning; 16 = Changeover; 32 = Maintenance; 64 = Break;
18. M07.WS_Cur_Mode: same as 14.)

## Confidentiality
The data is not confidential, feel free to upload to Google Colab etc.
