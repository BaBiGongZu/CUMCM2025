import ctypes
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc


# ==========================
# 常量定义（与其他题保持一致）
# ==========================
G_ACCEL = 9.8
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SPEED = 300.0
TARGET_FALSE = np.array([0.0, 0.0, 0.0])

VEC_M1_TO_TARGET = TARGET_FALSE - P_M1_0
DIST_M1_TO_TARGET = np.linalg.norm(VEC_M1_TO_TARGET)
UNIT_VEC_M1 = VEC_M1_TO_TARGET / DIST_M1_TO_TARGET
VEC_V_M1 = V_M1_SPEED * UNIT_VEC_M1
MISSILE_FLIGHT_TIME = DIST_M1_TO_TARGET / V_M1_SPEED

# FY1 无人机起点
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])

# 真目标参数（须与C库一致）
REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0

# 烟幕参数（须与C库一致）
R_SMOKE = 10.0
V_SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0

# 本题参数
NUM_GRENADES = 3
MIN_DROP_INTERVAL = 1.0

# 数值模拟步长（C判定每步）：越小越精细，越大越快
TIME_STEP = 0.01


# ==========================
# C 动态库封装（仅用到多云判定接口）
# ==========================
class SmokeBlockingCLibV2:
    """更轻量的C库封装：提供多云判定接口。"""

    def __init__(self, lib_dir: str = "libs"):
        self.lib_dir = lib_dir
        self.c_lib = None
        self._load_library()
        self._setup_function_interfaces()
        # 预热，避免并行时第一次初始化的竞争
        try:
            _ = self.get_key_points_count()
        except Exception:
            pass

    def _detect_platform(self) -> Tuple[str, str]:
        system = platform.system().lower()
        if system == "darwin":
            return "macos", ".dylib"
        elif system == "linux":
            return "linux", ".so"
        elif system == "windows":
            return "windows", ".dll"
        else:
            raise RuntimeError(f"不支持的平台: {system}")

    def _load_library(self):
        platform_name, extension = self._detect_platform()
        candidates = [
            os.path.join(self.lib_dir, f"smoke_blocking_{platform_name}{extension}"),
            os.path.join(self.lib_dir, f"smoke_blocking{extension}"),
            f"smoke_blocking_{platform_name}{extension}",
            f"smoke_blocking{extension}",
        ]
        last_err = None
        for p in candidates:
            if os.path.exists(p):
                try:
                    self.c_lib = ctypes.CDLL(p)
                    # 打印一次即可，子进程各自加载
                    print(f"✅ 已加载C库: {p}")
                    return
                except OSError as e:
                    last_err = e
        raise FileNotFoundError(f"未找到C库，请先编译: {candidates}\n最后错误: {last_err}")

    def _setup_function_interfaces(self):
        # bool check_multiple_clouds_blocking(const double missile_pos[3],
        #                                     const double cloud_positions[][3],
        #                                     int num_clouds)
        self.c_lib.check_multiple_clouds_blocking.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.c_lib.check_multiple_clouds_blocking.restype = ctypes.c_bool

        # double calculate_multiple_clouds_blocking_duration(const double missile_start[3],
        #                                                    const double missile_velocity[3],
        #                                                    const double explode_positions[][3],
        #                                                    int num_clouds,
        #                                                    double t_start,
        #                                                    double t_end,
        #                                                    double time_step,
        #                                                    double sink_speed)
        if hasattr(self.c_lib, "calculate_multiple_clouds_blocking_duration"):
            self.c_lib.calculate_multiple_clouds_blocking_duration.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
            ]
            self.c_lib.calculate_multiple_clouds_blocking_duration.restype = ctypes.c_double

        # 附带可选接口（调试/版本信息）
        if hasattr(self.c_lib, "get_version"):
            self.c_lib.get_version.restype = ctypes.c_char_p
        if hasattr(self.c_lib, "get_key_points_count"):
            self.c_lib.get_key_points_count.restype = ctypes.c_int

    def get_version(self) -> str:
        return self.c_lib.get_version().decode("utf-8") if hasattr(self.c_lib, "get_version") else "unknown"

    def get_key_points_count(self) -> int:
        return self.c_lib.get_key_points_count() if hasattr(self.c_lib, "get_key_points_count") else -1

    def check_multiple_clouds_blocking(self, missile_pos: np.ndarray, cloud_positions: np.ndarray) -> bool:
        """
        使用C库判定：当前时刻的多烟雾云是否对目标实现“完整遮蔽”。

        missile_pos: shape (3,)
        cloud_positions: shape (N, 3)
        """
        if cloud_positions.size == 0:
            return False

        missile_pos_c = (ctypes.c_double * 3)(*map(float, missile_pos))

        n = int(cloud_positions.shape[0])
        Cloud3 = ctypes.c_double * 3
        CloudArray = Cloud3 * n
        cloud_arr = CloudArray(*[Cloud3(*map(float, row)) for row in cloud_positions])

        # 重要：二维数组在ctypes中可以按平坦指针传入
        cloud_ptr = ctypes.cast(cloud_arr, ctypes.POINTER(ctypes.c_double))
        return bool(self.c_lib.check_multiple_clouds_blocking(missile_pos_c, cloud_ptr, ctypes.c_int(n)))

    def calculate_multiple_clouds_blocking_duration(
        self,
        t_start: float,
        t_end: float,
        cloud_positions_at_start: np.ndarray,
        time_step: float = TIME_STEP,
        sink_speed: float = V_SMOKE_SINK_SPEED,
    ) -> float:
        """
        在[t_start, t_end]区间内由C库积分遮蔽时长。
        注意：cloud_positions_at_start 应为各云团在 t_start 时刻的实际位置。
        """
        if t_start >= t_end:
            return 0.0

        n = int(cloud_positions_at_start.shape[0]) if cloud_positions_at_start is not None else 0
        if n == 0:
            return 0.0

        # 若无该函数，退化为逐步判定（慢但保证功能）
        if not hasattr(self.c_lib, "calculate_multiple_clouds_blocking_duration"):
            total = 0.0
            steps = int((t_end - t_start) / time_step) + 1
            for i in range(steps):
                t = t_start + i * time_step
                if t > t_end:
                    break
                missile_pos = P_M1_0 + VEC_V_M1 * t
                clouds = cloud_positions_at_start.copy()
                clouds[:, 2] = clouds[:, 2] - sink_speed * (t - t_start)
                clouds = clouds[clouds[:, 2] > 0.0]
                if clouds.size and self.check_multiple_clouds_blocking(missile_pos, clouds):
                    total += time_step
            return total

        missile_start_c = (ctypes.c_double * 3)(*map(float, P_M1_0))
        missile_vel_c = (ctypes.c_double * 3)(*map(float, VEC_V_M1))
        Cloud3 = ctypes.c_double * 3
        CloudArray = Cloud3 * n
        cloud_arr = CloudArray(*[Cloud3(*map(float, row)) for row in cloud_positions_at_start])
        cloud_ptr = ctypes.cast(cloud_arr, ctypes.POINTER(ctypes.c_double))

        return float(
            self.c_lib.calculate_multiple_clouds_blocking_duration(
                missile_start_c,
                missile_vel_c,
                cloud_ptr,
                ctypes.c_int(n),
                ctypes.c_double(t_start),
                ctypes.c_double(t_end),
                ctypes.c_double(time_step),
                ctypes.c_double(sink_speed),
            )
        )


# 模块级全局：便于多进程各自初始化
C_LIB = SmokeBlockingCLibV2()


# ==========================
# 业务逻辑：三枚烟雾弹参数化与目标函数
# ==========================
def uav_dir_from_angle(theta: float) -> np.ndarray:
    v = np.zeros(3)
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return v


@dataclass
class Grenade:
    t_drop: float
    t_delay: float
    explode_abs: float
    explode_pos: np.ndarray
    effective_end_abs: float  # 起爆后有效结束的绝对时刻（落地或20s，以较早者）


def build_grenades(uav_speed: float, theta: float, t_d1: float, dt12: float, dt23: float,
                   d1: float, d2: float, d3: float) -> List[Grenade]:
    dir3 = uav_dir_from_angle(theta)
    uav_v = uav_speed * dir3

    drops = [t_d1, t_d1 + dt12, t_d1 + dt12 + dt23]
    delays = [d1, d2, d3]

    result: List[Grenade] = []
    for t_drop, t_delay in zip(drops, delays):
        t_exp = t_drop + t_delay
        if t_exp >= MISSILE_FLIGHT_TIME:
            # 超过导弹飞行时间，无效，跳过
            continue

        # 投放点：FY1原点 + UAV匀速直线
        drop_pos = P_FY1_0 + uav_v * t_drop

        # 起爆点：继续以UAV速度平动 + 自由落体竖直下落
        explode_pos = drop_pos + uav_v * t_delay
        explode_pos = explode_pos.astype(float)
        explode_pos[2] -= 0.5 * G_ACCEL * (t_delay ** 2)

        # 起爆高度过低（<=目标顶面）视为无效
        if explode_pos[2] <= REAL_TARGET_HEIGHT:
            continue

        # 计算有效结束时刻：落地时刻或20秒窗口的较早者
        z0 = float(explode_pos[2])
        dt_land = z0 / V_SMOKE_SINK_SPEED if V_SMOKE_SINK_SPEED > 0 else SMOKE_DURATION
        eff_end_abs = t_exp + min(dt_land, SMOKE_DURATION)

        result.append(Grenade(
            t_drop=t_drop,
            t_delay=t_delay,
            explode_abs=t_exp,
            explode_pos=explode_pos,
            effective_end_abs=eff_end_abs,
        ))

    return result


def objective_3_grenades(params: np.ndarray) -> float:
    """
    参数: [uav_speed, theta, t_drop1, dt12, dt23, delay1, delay2, delay3]
    目标: 最大化有效遮蔽总时长（此函数返回其相反数以用于最小化）
    """
    uav_speed, theta, t_d1, dt12, dt23, d1, d2, d3 = params

    grenades = build_grenades(uav_speed, theta, t_d1, dt12, dt23, d1, d2, d3)
    if not grenades:
        return 0.0

    # 构造每枚弹的有效区间 [explode_abs, effective_end_abs]
    intervals: List[Tuple[float, float, int]] = []
    for idx, g in enumerate(grenades):
        s = g.explode_abs
        e = min(g.effective_end_abs, MISSILE_FLIGHT_TIME)
        if s < e:
            intervals.append((s, e, idx))

    if not intervals:
        return 0.0

    # 事件列表：开始/结束
    events: List[Tuple[float, str, int]] = []
    for s, e, i_idx in intervals:
        events.append((s, 'start', i_idx))
        events.append((e, 'end', i_idx))
    events.sort(key=lambda x: x[0])

    total_duration = 0.0
    active: set[int] = set()
    last_time: float | None = None

    for t_ev, ev_type, idx in events:
        # 对于上一个事件到本事件之间的区间，在活跃集合不变，调用C库积分
        if active and last_time is not None and t_ev > last_time:
            clouds0: List[np.ndarray] = []
            for i_idx in active:
                g = grenades[i_idx]
                # 在 last_time 时刻的云团位置
                dt0 = last_time - g.explode_abs
                if 0.0 <= dt0 <= SMOKE_DURATION:
                    pos0 = g.explode_pos.copy()
                    pos0[2] -= V_SMOKE_SINK_SPEED * dt0
                    if pos0[2] > 0.0:
                        clouds0.append(pos0)

            if clouds0:
                cloud_mat = np.vstack(clouds0)
                total_duration += C_LIB.calculate_multiple_clouds_blocking_duration(
                    t_start=last_time,
                    t_end=t_ev,
                    cloud_positions_at_start=cloud_mat,
                    time_step=TIME_STEP,
                    sink_speed=V_SMOKE_SINK_SPEED,
                )

        if ev_type == 'start':
            active.add(idx)
        else:
            active.discard(idx)

        last_time = t_ev

    return -total_duration


# ==========================
# 可执行主流程：差分进化 + 并行
# ==========================
def run_optimization():
    print("🎯 问题三（C库加速，多进程）：3枚烟雾弹协同优化")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s; 时间步长: {TIME_STEP:.3f} s")
    print(f"C库版本: {C_LIB.get_version()} | 关键点: {C_LIB.get_key_points_count()}")

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 变量边界
    bounds = [
        (70.0, 140.0),      # uav_speed
        (0.0, 2 * math.pi), # theta
        (0.0, 10.0),        # t_drop1
        (MIN_DROP_INTERVAL, 12.0),  # dt12
        (MIN_DROP_INTERVAL, 12.0),  # dt23
        (0.0, 20.0),        # delay1
        (0.0, 20.0),        # delay2
        (0.0, 20.0),        # delay3
    ]

    # 多组启发式种子（参数顺序：uav_speed, theta, t_drop1, dt12, dt23, delay1, delay2, delay3）
    heuristic_seeds = np.array([
        [130.0, 0.10, 0.5, 1.2, 1.2, 2.0, 3.0, 4.0],   # 合理猜测
        [120.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.5, 4.0],     # 均匀时间分布
        [110.0, 0.1, 1.5, 2.5, 2.0, 2.5, 3.0, 3.5],     # 略慢速度
        [130.0, -0.1, 0.5, 1.5, 1.5, 4.0, 2.8, 4.2],    # 早投高延迟
        [125.0, 0.05, 1.0, 1.0, 1.0, 2.0, 2.5, 3.0],    # 快速连发（间隔=1s）
        [115.0, 0.0, 2.0, 2.0, 2.0, 5.0, 5.5, 6.0],     # 延迟爆炸为主
        [105.0, 0.2, 0.5, 1.0, 1.0, 1.5, 1.8, 2.0],     # 紧密短延迟
        [135.0, -0.2, 2.5, 3.0, 3.0, 3.0, 4.0, 3.5],    # 分散时间分布
        [116.399481495084, 0.117001154452, 0.000001857460, 1.002586950324, 1.853282128725, 0.001052852045, 0.005464179587, 0.443985076076], # 手动优化候选
        [139.976844, 3.135471, 0.001625, 3.656653, 1.919484, 3.609524, 5.341602, 6.039781],
    ])

    # 生成初始种群（启发式种子 + 拉丁超立方）
    TOTAL_POP = 1500
    num_vars = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    sampler = qmc.LatinHypercube(d=num_vars, seed=42)
    n_random = max(0, TOTAL_POP - len(heuristic_seeds))
    rand_init = sampler.random(n=n_random)
    rand_scaled = qmc.scale(rand_init, lb, ub) if n_random > 0 else np.empty((0, num_vars))
    init_pop = np.vstack([heuristic_seeds, rand_scaled])

    print(f"初始种群: {init_pop.shape[0]} 个体（含启发式种子 {len(heuristic_seeds)}），变量数: {num_vars}")
    print("开始并行优化（workers=-1 使用多进程）...")

    t0 = time.time()
    result = differential_evolution(
        objective_3_grenades,
        bounds,
        init=init_pop,
        strategy="rand1bin",
        maxiter=3000,
        popsize=500,
        tol=0.01,
        recombination=0.8,
        mutation=(0.6, 1.5),
        disp=True,
        workers=-1,  # 多进程并行
        seed=42,
    )
    t1 = time.time()
    print(f"\n优化完成，总耗时: {t1 - t0:.2f} s")

    if result.success and result.fun < -1e-9:
        best = result.x
        duration = -result.fun
        print_solution(best, duration)

        print("\n======== 下一轮迭代种子 ========")
        print(
            "seed = np.array([[\n    "
            + ", ".join(f"{v:.12f}" for v in best)
            + "\n]])"
        )
    else:
        print("❌ 优化失败或未找到有效解")
        if result.fun is not None:
            print(f"最佳找到值: {-result.fun:.6f} 秒")


def print_solution(params: np.ndarray, duration: float):
    uav_speed, theta, t_d1, dt12, dt23, d1, d2, d3 = params
    dir3 = uav_dir_from_angle(theta)
    uav_v = uav_speed * dir3
    drops = [t_d1, t_d1 + dt12, t_d1 + dt12 + dt23]
    delays = [d1, d2, d3]

    print("\n最优策略（C库加速，多云判定）：")
    print(f"  > 最大有效遮蔽: {duration:.6f} s")
    print("-" * 60)
    print(f"UAV 速度: {uav_speed:.3f} m/s | 航向角: {theta:.6f} rad ({math.degrees(theta):.2f}°)")
    for i in range(NUM_GRENADES):
        t_drop = drops[i]
        t_delay = delays[i]
        t_exp = t_drop + t_delay
        drop_pos = P_FY1_0 + uav_v * t_drop
        explode_pos = drop_pos + uav_v * t_delay
        explode_pos = explode_pos.astype(float)
        explode_pos[2] -= 0.5 * G_ACCEL * (t_delay ** 2)
        print(f"弹{i+1}: 投放@{t_drop:.3f}s, 延时{t_delay:.3f}s, 起爆@{t_exp:.3f}s")
        print(
            f"   投放点=({drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f}) | "
            f"起爆点=({explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f})"
        )

    # 详细报告：每枚弹独立遮蔽时长与时间窗口、时间重叠分析
    grenades = build_grenades(uav_speed, theta, t_d1, dt12, dt23, d1, d2, d3)
    if grenades:
        print("-" * 60)
        print("单枚烟雾弹遮蔽时长（各自独立作用）:")
        total_start = min(g.explode_abs for g in grenades)
        total_end = min(max(g.effective_end_abs for g in grenades), MISSILE_FLIGHT_TIME)
        for idx, g in enumerate(grenades, 1):
            start_i = g.explode_abs
            end_i = min(g.effective_end_abs, MISSILE_FLIGHT_TIME)
            pos0 = g.explode_pos.reshape(1, 3)
            dur_i = C_LIB.calculate_multiple_clouds_blocking_duration(
                t_start=start_i,
                t_end=end_i,
                cloud_positions_at_start=pos0,
                time_step=TIME_STEP,
                sink_speed=V_SMOKE_SINK_SPEED,
            )
            print(
                f"  第{idx}枚: 窗口[{start_i:.3f}s, {end_i:.3f}s], 独立遮蔽时长: {dur_i:.6f}s"
            )

        # 时间重叠（按有效窗口的时间交集，仅基于时间窗，不代表遮蔽重叠）
        print("-" * 60)
        print("时间窗口重叠分析（仅时间窗交集）:")
        intervals = [(idx + 1, g.explode_abs, min(g.effective_end_abs, MISSILE_FLIGHT_TIME)) for idx, g in enumerate(grenades)]
        intervals.sort(key=lambda x: x[1])
        for i in range(len(intervals)):
            id_i, s_i, e_i = intervals[i]
            overlaps = []
            for j in range(len(intervals)):
                if i == j:
                    continue
                id_j, s_j, e_j = intervals[j]
                os_ = max(s_i, s_j)
                oe_ = min(e_i, e_j)
                if os_ < oe_:
                    overlaps.append((id_j, os_, oe_))
            if overlaps:
                msg = ", ".join([f"与第{oid}枚[{os_:.3f},{oe_:.3f}]" for oid, os_, oe_ in overlaps])
            else:
                msg = "无"
            print(f"  第{id_i}枚窗口[{s_i:.3f},{e_i:.3f}] 重叠: {msg}")


def quick_sanity_check():
    """快速自检：使用题目1的单弹参数在多弹框架下跑一小步。"""
    print("\n🧪 快速自检...")
    params = np.array([120.0, 0.1, 1.5, 2.0, 2.0, 3.6, 100.0, 100.0])
    val = objective_3_grenades(params)
    print(f"目标函数返回(负遮蔽时长): {val:.6f}")


if __name__ == "__main__":
    # 轻量自检，随后启动优化（可按需注释）
    #quick_sanity_check()
    run_optimization()
