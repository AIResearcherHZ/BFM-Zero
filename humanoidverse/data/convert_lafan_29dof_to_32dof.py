"""将29DOF(G1)的lafan motion数据转换为32DOF(Taks_T1)格式。
Taks_T1比G1多3个neck关节(neck_yaw, neck_roll, neck_pitch)，在末尾追加零值。

用法:
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof \
        --input data/lafan_29dof.pkl \
        --output data/lafan_32dof.pkl

    # 或者转换10s-clipped版本:
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof \
        --input humanoidverse/data/lafan_29dof_10s-clipped.pkl \
        --output humanoidverse/data/lafan_32dof_10s-clipped.pkl
"""
import argparse
import pickle
import numpy as np
from pathlib import Path


# G1 29DOF关节顺序:
# left_leg(6) + right_leg(6) + waist(2) + left_arm(7) + right_arm(7) + head(1) = 29
# Taks_T1 32DOF关节顺序:
# left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7) + neck(3) = 32
#
# 映射策略: 在29DOF数据末尾补3个零值(neck_yaw, neck_roll, neck_pitch)
# 注: waist多出的1个DOF(waist_pitch)也需要处理
# G1 waist: waist_yaw(12), waist_roll(13)
# Taks_T1 waist: waist_yaw(12), waist_roll(13), waist_pitch(14)
# G1 torso_link(head): index 14
# 实际映射:
#   G1[0:14] -> Taks_T1[0:14]   (legs + waist_yaw + waist_roll)
#   插入 waist_pitch = 0          (Taks_T1 index 14)
#   G1[14:29] -> Taks_T1[15:30]  (arms)
#   插入 neck(3) = [0,0,0]        (Taks_T1 index 30,31,32... wait no index 29,30,31)

# 实际上需要仔细看G1的29个关节:
# G1 29DOF:
# left_hip_pitch(0), left_hip_roll(1), left_hip_yaw(2), left_knee(3), left_ankle_pitch(4), left_ankle_roll(5)
# right_hip_pitch(6), right_hip_roll(7), right_hip_yaw(8), right_knee(9), right_ankle_pitch(10), right_ankle_roll(11)
# waist_yaw(12), waist_roll(13)  [2 waist joints]
# torso/waist_pitch(14) ← G1只有这个是head的, 但实际上G1这里是...
# 让我重新看G1的关节列表

# 根据g1_29dof.yaml的dof_names:
# G1 29DOF: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
#           right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
#           waist_yaw, waist_roll, waist_pitch,  [3 waist joints]
#           left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
#           left_wrist_roll, left_wrist_pitch, left_wrist_yaw,  [7 left arm]
#           right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
#           right_wrist_roll, right_wrist_pitch, right_wrist_yaw  [7 right arm]
# = 6+6+3+7+7 = 29

# Taks_T1 32DOF:
# 同G1的29个 + neck_yaw(29), neck_roll(30), neck_pitch(31)
# = 29 + 3 = 32

# 所以映射非常简单: 在末尾补3个零值!

NECK_DOF = 3  # neck_yaw, neck_roll, neck_pitch


def pad_array(arr, axis=-1):
    """在最后一个关节维度末尾补3个零值"""
    if arr is None:
        return None
    shape = list(arr.shape)
    pad_shape = list(arr.shape)
    pad_shape[axis] = NECK_DOF
    padding = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, padding], axis=axis)


def convert_motion_dict(motion):
    """转换单个motion数据"""
    result = {}
    for key, val in motion.items():
        if isinstance(val, np.ndarray):
            # 只对关节维度的数据做padding
            if val.ndim >= 1 and val.shape[-1] == 29:
                result[key] = pad_array(val, axis=-1)
            elif val.ndim >= 1 and val.shape[-1] == 36:
                # 36 = 7(freejoint qpos) + 29(joints), pad joints部分
                free = val[..., :7]
                joints = val[..., 7:]
                padded_joints = pad_array(joints, axis=-1)
                result[key] = np.concatenate([free, padded_joints], axis=-1)
            elif val.ndim >= 1 and val.shape[-1] == 35:
                # 35 = 6(freejoint qvel) + 29(joints), pad joints部分
                free = val[..., :6]
                joints = val[..., 6:]
                padded_joints = pad_array(joints, axis=-1)
                result[key] = np.concatenate([free, padded_joints], axis=-1)
            else:
                result[key] = val
        elif isinstance(val, dict):
            result[key] = convert_motion_dict(val)
        else:
            result[key] = val
    return result


def convert_file(input_path: str, output_path: str):
    print(f"读取: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"数据类型: {type(data)}")
    if isinstance(data, dict):
        converted = convert_motion_dict(data)
        # 打印转换信息
        for key in list(converted.keys())[:5]:
            val = converted[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: {val.shape}")
    elif isinstance(data, list):
        converted = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                converted.append(convert_motion_dict(item))
            else:
                converted.append(item)
            if i == 0:
                for key in list(converted[0].keys())[:5]:
                    val = converted[0][key]
                    if isinstance(val, np.ndarray):
                        print(f"  {key}: {val.shape}")
        print(f"  共 {len(converted)} 个motion片段")
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(converted, f)
    print(f"写入: {output_path}")
    print("完成！29DOF → 32DOF（末尾补3个neck零值）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将29DOF lafan motion数据转换为32DOF")
    parser.add_argument("--input", required=True, help="输入pkl文件路径")
    parser.add_argument("--output", required=True, help="输出pkl文件路径")
    args = parser.parse_args()
    convert_file(args.input, args.output)
