"""
与 solve3_lib.py 完全一致的参数化计算程序
用于验证三枚烟幕弹优化结果的正确性
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Solve3CompatibleInterference:
    def __init__(self, uav_speed=139.976844, flight_angle=3.135471,
                 t_drop1=0.001625, t_interval2=3.656653, t_interval3=1.919484,
                 t_delay1=3.609524, t_delay2=5.341602, t_delay3=6.039781):
        """
        使用与 solve3_lib.py 完全一致的物理模型

        Parameters:
        -----------
        uav_speed : float
            无人机飞行速度 (m/s)
        flight_angle : float
            飞行角度（弧度）
        t_drop1 : float
            第一枚烟幕弹投放时间 (s)
        t_interval2 : float
            第二枚烟幕弹投放间隔 (s)
        t_interval3 : float
            第三枚烟幕弹投放间隔 (s)
        t_delay1, t_delay2, t_delay3 : float
            各烟幕弹投放后到起爆的延迟时间 (s)
        """
        print("初始化与solve3_lib.py一致的三枚烟幕弹干扰问题...")
        print(f"  - 无人机速度: {uav_speed:.6f} m/s")
        print(f"  - 飞行角度: {flight_angle:.6f} 弧度 ({math.degrees(flight_angle):.2f}°)")
        print(f"  - 投放时间: {t_drop1:.6f}s, {t_drop1 + t_interval2:.6f}s, {t_drop1 + t_interval2 + t_interval3:.6f}s")
        print(f"  - 起爆延迟: {t_delay1:.6f}s, {t_delay2:.6f}s, {t_delay3:.6f}s")

        # 参数设置（与solve3_lib.py一致）
        self.uav_speed = uav_speed
        self.flight_angle = flight_angle
        self.t_drop1 = t_drop1
        self.t_interval2 = t_interval2
        self.t_interval3 = t_interval3
        self.t_delay1 = t_delay1
        self.t_delay2 = t_delay2
        self.t_delay3 = t_delay3

        # 场景参数（与solve3_lib.py完全一致）
        self.fake_target = np.array([0.0, 0.0, 0.0])
        self.real_target_center = np.array([0.0, 200.0, 0.0])
        self.real_target_radius = 7.0
        self.real_target_height = 10.0

        # 导弹参数（与solve3_lib.py完全一致）
        self.missile_pos = np.array([20000.0, 0.0, 2000.0])
        self.missile_speed = 300.0
        self.missile_target = self.fake_target
        self.missile_distance = np.linalg.norm(self.missile_pos - self.missile_target)
        self.missile_direction = (self.missile_target - self.missile_pos) / self.missile_distance
        self.missile_flight_time = self.missile_distance / self.missile_speed

        # 无人机参数（与solve3_lib.py完全一致）
        self.uav_pos = np.array([17800.0, 0.0, 1800.0])

        # 烟雾参数（与solve3_lib.py完全一致）
        self.gravity = 9.8
        self.cloud_sink_speed = 3.0
        self.effective_radius = 10.0
        self.effective_duration = 20.0

        # 生成目标关键点（与solve3_lib.py完全一致）
        self.target_key_points = self._generate_target_key_points()

        # 计算绝对投放时间
        self.absolute_drop_times = [
            self.t_drop1,
            self.t_drop1 + self.t_interval2,
            self.t_drop1 + self.t_interval2 + self.t_interval3
        ]
        self.delays = [self.t_delay1, self.t_delay2, self.t_delay3]

        print(f"  - 导弹飞行总时间: {self.missile_flight_time:.2f} s")
        print(f"  - 目标关键点数量: {len(self.target_key_points)}")
        print(f"  - 绝对投放时间: {[f'{t:.6f}' for t in self.absolute_drop_times]}")

    def _generate_target_key_points(self, num_points_per_circle=50):
        """生成真目标圆柱体上的关键点（与solve3_lib.py完全一致）"""
        key_points = []

        angles = np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False)

        # 底面圆（z=0）
        for angle in angles:
            x = self.real_target_center[0] + self.real_target_radius * np.cos(angle)
            y = self.real_target_center[1] + self.real_target_radius * np.sin(angle)
            z = 0.0
            key_points.append([x, y, z])

        # 顶面圆（z=10）
        for angle in angles:
            x = self.real_target_center[0] + self.real_target_radius * np.cos(angle)
            y = self.real_target_center[1] + self.real_target_radius * np.sin(angle)
            z = self.real_target_height
            key_points.append([x, y, z])

        return np.array(key_points)

    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离（与solve3_lib.py完全一致）"""
        line_vec = line_end - line_start
        line_length_sq = np.dot(line_vec, line_vec)

        if line_length_sq < 1e-10:
            return np.linalg.norm(point - line_start)

        point_vec = point - line_start
        projection = np.dot(point_vec, line_vec) / line_length_sq

        if projection < 0:
            closest_point = line_start
        elif projection > 1:
            closest_point = line_end
        else:
            closest_point = line_start + projection * line_vec

        return np.linalg.norm(point - closest_point)

    def calculate_uav_direction(self):
        """计算无人机飞行方向（与solve3_lib.py逻辑一致）"""
        uav_direction = np.zeros(3)
        uav_direction[0] = np.cos(self.flight_angle)
        uav_direction[1] = np.sin(self.flight_angle)
        return uav_direction

    def calculate_smoke_trajectories(self):
        """计算三枚烟幕弹的轨迹"""
        uav_direction = self.calculate_uav_direction()
        trajectories = []

        for i in range(3):
            t_drop = self.absolute_drop_times[i]
            t_delay = self.delays[i]

            # 投放位置
            drop_pos = self.uav_pos + self.uav_speed * t_drop * uav_direction

            # 起爆位置
            bomb_initial_velocity = self.uav_speed * uav_direction
            explode_pos = drop_pos.copy()
            explode_pos += bomb_initial_velocity * t_delay
            explode_pos[2] -= 0.5 * self.gravity * t_delay**2

            t_explode_abs = t_drop + t_delay

            trajectories.append({
                'bomb_id': i + 1,
                'drop_pos': drop_pos,
                'explode_pos': explode_pos,
                't_drop': t_drop,
                't_delay': t_delay,
                't_explode': t_explode_abs
            })

        return trajectories

    def calculate_total_blocking_duration(self):
        """计算总遮蔽时长（与solve3_lib.py完全一致的算法）"""
        # 1. 计算烟幕弹轨迹
        trajectories = self.calculate_smoke_trajectories()

        # 2. 检查约束条件
        for traj in trajectories:
            if traj['t_explode'] >= self.missile_flight_time:
                return 0.0, f"烟幕弹{traj['bomb_id']}在导弹到达后起爆", trajectories

            if traj['explode_pos'][2] <= self.real_target_height:
                return 0.0, f"烟幕弹{traj['bomb_id']}在目标高度以下起爆", trajectories

        # 3. 模拟遮蔽过程
        print("开始模拟三枚烟幕弹遮蔽过程...")
        dt = 0.01
        total_blocked_time = 0

        # 确定模拟时间范围
        earliest_explode = min(traj['t_explode'] for traj in trajectories)
        latest_end = max(traj['t_explode'] + self.effective_duration for traj in trajectories)
        sim_end_time = min(latest_end, self.missile_flight_time)

        print(f"  - 模拟时间范围: {earliest_explode:.3f} - {sim_end_time:.3f} s")

        step_count = 0
        total_steps = int((sim_end_time - earliest_explode) / dt)

        for t in np.arange(earliest_explode, sim_end_time, dt):
            if t > self.missile_flight_time:
                break

            # 当前导弹位置
            current_missile_pos = self.missile_pos + self.missile_speed * t * self.missile_direction

            # 检查每个烟雾云的遮蔽效果
            all_blocked = True
            blocked_count = 0
            active_clouds = 0

            for key_point in self.target_key_points:
                point_blocked = False

                # 检查所有活跃的烟雾云
                for traj in trajectories:
                    if (traj['t_explode'] <= t <= traj['t_explode'] + self.effective_duration):
                        active_clouds += 1
                        time_since_explode = t - traj['t_explode']
                        current_cloud_pos = traj['explode_pos'].copy()
                        current_cloud_pos[2] -= self.cloud_sink_speed * time_since_explode

                        if current_cloud_pos[2] > 0:  # 烟雾还在空中
                            dist = self.point_to_line_distance(
                                current_cloud_pos, current_missile_pos, key_point
                            )
                            if dist <= self.effective_radius:
                                point_blocked = True
                                break

                if point_blocked:
                    blocked_count += 1
                else:
                    all_blocked = False

            if all_blocked and blocked_count > 0:
                total_blocked_time += dt

            step_count += 1
            if step_count % 1000 == 0:
                progress = step_count / total_steps * 100
                print(f"  - 进度: {progress:.1f}% (遮蔽点: {blocked_count}/{len(self.target_key_points)}, 活跃云: {active_clouds})")

        return total_blocked_time, "计算完成", trajectories

    def get_detailed_results(self):
        """获取详细的计算结果"""
        duration, status, trajectories = self.calculate_total_blocking_duration()

        return {
            'total_blocking_duration': duration,
            'status': status,
            'trajectories': trajectories,
            'uav_speed': self.uav_speed,
            'flight_angle_deg': math.degrees(self.flight_angle),
            'missile_flight_time': self.missile_flight_time
        }


def main():
    """主函数"""
    print("🎯 与solve3_lib.py一致的三枚烟幕弹验证程序")
    print("="*80)

    # 使用solve3_lib.py的最优参数（示例）
    problem = Solve3CompatibleInterference(
        uav_speed=139.976844,
        flight_angle=3.135471,
        t_drop1=0.001625,
        t_interval2=3.656653,
        t_interval3=1.919484,
        t_delay1=3.609524,
        t_delay2=5.341602,
        t_delay3=6.039781
    )

    results = problem.get_detailed_results()

    print("\n" + "="*25 + " 验证结果 " + "="*25)
    print(f"✅ 总有效遮蔽时长: {results['total_blocking_duration']:.6f} 秒")
    print(f"📊 计算状态: {results['status']}")
    print(f"🚁 无人机速度: {results['uav_speed']:.2f} m/s")
    print(f"🧭 飞行角度: {results['flight_angle_deg']:.2f}°")
    print(f"⏱️  导弹飞行时间: {results['missile_flight_time']:.2f} s")

    print("\n" + "="*25 + " 烟幕弹详情 " + "="*25)
    for traj in results['trajectories']:
        print(f"烟幕弹 {traj['bomb_id']}:")
        print(f"  📍 投放时间: {traj['t_drop']:.6f} s")
        print(f"  💥 起爆时间: {traj['t_explode']:.6f} s")
        print(f"  📍 投放位置: [{traj['drop_pos'][0]:.1f}, {traj['drop_pos'][1]:.1f}, {traj['drop_pos'][2]:.1f}]")
        print(f"  💥 起爆位置: [{traj['explode_pos'][0]:.1f}, {traj['explode_pos'][1]:.1f}, {traj['explode_pos'][2]:.1f}]")
        print()

    print("="*80)
    print("\n注意：此结果应该与solve3_lib.py的优化结果完全一致！")
    print("如果存在差异，请检查参数设置或算法实现。")


if __name__ == "__main__":
    main()

