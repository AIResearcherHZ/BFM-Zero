"""将29DOF(G1)的lafan motion数据转换为32DOF(Taks_T1)格式。

G1 29DOF = left_leg(6)+right_leg(6)+waist(3)+left_arm(7)+right_arm(7)
Taks_T1 32DOF = 同G1 + neck(3): neck_yaw, neck_roll, neck_pitch

映射策略: 在末尾补3个零值（neck关节初始为0）

用法:
    # 转换单个文件
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof \\
        --input humanoidverse/data/lafan_29dof.pkl \\
        --output humanoidverse/data/lafan_32dof.pkl

    # 自动转换两个标准文件到 data/ 目录
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof --auto

    # 转换并可视化验证（对比29dof与32dof在MuJoCo中的运动）
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof --auto --visualize

    # 仅验证已有的32dof文件格式是否正确
    python -m humanoidverse.data.convert_lafan_29dof_to_32dof \\
        --validate humanoidverse/data/lafan_29dof.pkl humanoidverse/data/lafan_32dof.pkl
"""
import argparse
import sys
import numpy as np
from pathlib import Path
try:
    import joblib as _loader
except ImportError:
    import pickle as _loader  # fallback

NECK_DOF = 3  # neck_yaw, neck_roll, neck_pitch


def pad_array(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """在指定轴末尾补NECK_DOF个零值"""
    pad_shape = list(arr.shape)
    pad_shape[axis] = NECK_DOF
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=axis)


def convert_motion_dict(motion: dict) -> dict:
    """递归转换单个motion dict，对所有末维为29/36/35的ndarray做padding"""
    result = {}
    for key, val in motion.items():
        if isinstance(val, np.ndarray):
            if val.shape[-1] == 29:
                result[key] = pad_array(val)
            elif val.shape[-1] == 36:
                # 36 = 7(freejoint qpos) + 29(joints)
                result[key] = np.concatenate([val[..., :7], pad_array(val[..., 7:])], axis=-1)
            elif val.shape[-1] == 35:
                # 35 = 6(freejoint qvel) + 29(joints)
                result[key] = np.concatenate([val[..., :6], pad_array(val[..., 6:])], axis=-1)
            else:
                result[key] = val
        elif isinstance(val, dict):
            result[key] = convert_motion_dict(val)
        else:
            result[key] = val
    return result


def _load(path):
    """自动用joblib或pickle加载"""
    try:
        import joblib
        return joblib.load(str(path))
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def _save(obj, path):
    """自动用joblib或pickle保存"""
    try:
        import joblib
        joblib.dump(obj, str(path))
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def convert_file(input_path: str, output_path: str, verbose: bool = True) -> None:
    """转换pkl文件：29DOF → 32DOF"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if verbose:
        print(f"[转换] 读取: {input_path}")
    data = _load(input_path)

    if isinstance(data, list):
        # list of motion dicts
        converted = [convert_motion_dict(item) if isinstance(item, dict) else item for item in data]
        if verbose:
            print(f"  list格式，共 {len(converted)} 个片段")
            _show_shapes(converted[0])
    elif isinstance(data, dict):
        # dict: key=motion_name, value=motion_dict
        converted = {k: convert_motion_dict(v) if isinstance(v, dict) else v for k, v in data.items()}
        if verbose:
            print(f"  dict格式，共 {len(converted)} 个motion")
            first_key = next(iter(converted))
            _show_shapes(converted[first_key])
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save(converted, output_path)
    if verbose:
        print(f"[转换] 写入: {output_path}  ✓ 29DOF→32DOF完成")


def _show_shapes(motion: dict, max_keys: int = 8) -> None:
    """打印motion dict中前几个ndarray的shape"""
    count = 0
    for key, val in motion.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape} dtype={val.dtype}")
            count += 1
            if count >= max_keys:
                break


def validate_alignment(path_29dof: str, path_32dof: str) -> bool:
    """验证32dof数据与29dof数据对齐（数值+维度检查）"""
    print(f"\n[验证] 对比 29DOF({path_29dof}) vs 32DOF({path_32dof})")
    data29 = _load(path_29dof)
    data32 = _load(path_32dof)

    # 统一为list（dict格式取values）
    items29 = data29 if isinstance(data29, list) else list(data29.values())
    items32 = data32 if isinstance(data32, list) else list(data32.values())

    if len(items29) != len(items32):
        print(f"  [ERROR] 片段数不一致: 29dof={len(items29)}, 32dof={len(items32)}")
        return False

    all_ok = True
    sample_idx = min(3, len(items29))
    for i in range(sample_idx):
        m29, m32 = items29[i], items32[i]
        for key in m29:
            if key not in m32:
                print(f"  [ERROR] 片段{i} 缺少key: {key}")
                all_ok = False
                continue
            v29, v32 = m29[key], m32[key]
            if not isinstance(v29, np.ndarray):
                continue
            # 检查维度扩展是否正确
            if v29.shape[-1] == 29:
                assert v32.shape[-1] == 32, f"片段{i} key={key}: 期望32维，得{v32.shape[-1]}"
                # 前29维应与原始数据一致
                if not np.allclose(v32[..., :29], v29, atol=1e-6):
                    print(f"  [ERROR] 片段{i} key={key}: 前29维不匹配!")
                    all_ok = False
                    continue
                # 后3维应为0
                if not np.allclose(v32[..., 29:], 0, atol=1e-6):
                    print(f"  [WARN] 片段{i} key={key}: neck(后3维)非零: {v32[..., 29:].max():.4f}")
                print(f"  [OK] 片段{i} key={key}: {v29.shape}→{v32.shape} ✓")
            elif v29.shape[-1] == 36:
                assert v32.shape[-1] == 39, f"片段{i} key={key}: 期望39维，得{v32.shape[-1]}"
                if not np.allclose(v32[..., :7], v29[..., :7], atol=1e-6):
                    print(f"  [ERROR] 片段{i} key={key}: freejoint部分不匹配!")
                    all_ok = False
                print(f"  [OK] 片段{i} key={key}: {v29.shape}→{v32.shape} ✓")

    if all_ok:
        print("[验证] 所有检查通过 ✓")
    else:
        print("[验证] 存在错误，请检查上方输出")
    return all_ok


def visualize_comparison(path_29dof: str, path_32dof: str, num_frames: int = 100) -> None:
    """用MuJoCo并排可视化G1(29dof)和Taks_T1(32dof)的运动数据对比"""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("[可视化] 需要安装 mujoco 包: pip install mujoco")
        return

    from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
    from humanoidverse.utils.taks_t1_env_config import get_taks_t1_robot_xml_root

    g1_xml = str(get_g1_robot_xml_root() / "scene_29dof_freebase_mujoco.xml")
    t1_xml = str(get_taks_t1_robot_xml_root() / "scene_Taks_T1.xml")

    print(f"[可视化] 加载 G1 模型: {g1_xml}")
    print(f"[可视化] 加载 Taks_T1 模型: {t1_xml}")

    data29 = _load(path_29dof)
    data32 = _load(path_32dof)

    items29 = data29 if isinstance(data29, list) else list(data29.values())
    items32 = data32 if isinstance(data32, list) else list(data32.values())

    # 取第一个片段的qpos序列
    m29, m32 = items29[0], items32[0]
    # 尝试找qpos字段
    qpos29_key = next((k for k in m29 if "qpos" in k.lower() and isinstance(m29[k], np.ndarray) and m29[k].shape[-1] == 36), None)
    qpos32_key = next((k for k in m32 if "qpos" in k.lower() and isinstance(m32[k], np.ndarray) and m32[k].shape[-1] == 39), None)

    if qpos29_key is None or qpos32_key is None:
        print(f"[可视化] 未找到qpos字段（已找到key: {list(m29.keys())[:6]}...）")
        print("[可视化] 尝试打印所有ndarray shape:")
        for k, v in m29.items():
            if isinstance(v, np.ndarray):
                print(f"  29dof {k}: {v.shape}")
        return

    qpos29 = m29[qpos29_key][:num_frames]
    qpos32 = m32[qpos32_key][:num_frames]
    T = min(len(qpos29), len(qpos32))
    print(f"[可视化] 使用字段 '{qpos29_key}', 共 {T} 帧")

    model29 = mujoco.MjModel.from_xml_path(g1_xml)
    model32 = mujoco.MjModel.from_xml_path(t1_xml)
    data29m = mujoco.MjData(model29)
    data32m = mujoco.MjData(model32)

    print("[可视化] 启动 MuJoCo viewer (左=G1 29dof, 右=Taks_T1 32dof)")
    print("  按 Esc 退出")

    with mujoco.viewer.launch_passive(model29, data29m) as v29, \
         mujoco.viewer.launch_passive(model32, data32m) as v32:
        for i in range(T):
            q29 = qpos29[i]
            q32 = qpos32[i]
            if len(q29) == model29.nq:
                data29m.qpos[:] = q29
            if len(q32) == model32.nq:
                data32m.qpos[:] = q32
            mujoco.mj_forward(model29, data29m)
            mujoco.mj_forward(model32, data32m)
            v29.sync()
            v32.sync()
            if not v29.is_running() or not v32.is_running():
                break

    print("[可视化] 结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将29DOF lafan motion数据转换为32DOF(Taks_T1格式)")
    parser.add_argument("--input", default=None, help="输入29dof pkl文件路径")
    parser.add_argument("--output", default=None, help="输出32dof pkl文件路径")
    parser.add_argument("--auto", action="store_true",
                        help="自动转换 lafan_29dof.pkl 和 lafan_29dof_10s-clipped.pkl")
    parser.add_argument("--validate", nargs=2, metavar=("29DOF", "32DOF"),
                        help="验证已有32dof文件与29dof对齐")
    parser.add_argument("--visualize", action="store_true",
                        help="MuJoCo可视化对比（需配合--auto或--input/--output）")
    parser.add_argument("--frames", type=int, default=200, help="可视化帧数(默认200)")
    args = parser.parse_args()

    # 推断数据根目录
    _here = Path(__file__).resolve().parent
    _data_dir = _here

    if args.validate:
        validate_alignment(args.validate[0], args.validate[1])
        sys.exit(0)

    pairs = []
    if args.auto:
        pairs = [
            (_data_dir / "lafan_29dof.pkl", _data_dir / "lafan_32dof.pkl"),
            (_data_dir / "lafan_29dof_10s-clipped.pkl", _data_dir / "lafan_32dof_10s-clipped.pkl"),
        ]
    elif args.input and args.output:
        pairs = [(Path(args.input), Path(args.output))]
    else:
        parser.print_help()
        sys.exit(1)

    for inp, out in pairs:
        if not inp.exists():
            print(f"[跳过] 输入文件不存在: {inp}")
            continue
        convert_file(str(inp), str(out))
        if args.validate or True:
            validate_alignment(str(inp), str(out))

    if args.visualize and pairs:
        inp, out = pairs[0]
        if inp.exists() and out.exists():
            visualize_comparison(str(inp), str(out), num_frames=args.frames)