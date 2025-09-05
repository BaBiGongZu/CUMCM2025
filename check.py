"""
与 solve1.py 完全一致的参数化计算程序
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Solve1CompatibleInterference:
    def __init__(self, uav_speed=120.0, flight_angle_offset=0.0, 
                 t_drop=1.5, t_explode_delay=3.6):
        """
        使用与 solve1.py 完全一致的物理模型
        
        Parameters:
        -----------
        uav_speed : float
            无人机飞行速度 (m/s)
        flight_angle_offset : float
            相对于朝向假目标方向的角度偏移（弧度）
        t_drop : float
            受领任务后的投放时间 (s)
        t_explode_delay : float
            投放后到起爆的延迟时间 (s)
        """
        print("初始化与solve1.py一致的参数化干扰问题...")
        print(f"  - 无人机速度: {uav_speed} m/s")
        print(f"  - 飞行角度偏移: {flight_angle_offset:.6f} 弧度 ({math.degrees(flight_angle_offset):.2f}°)")
        print(f"  - 投放时间: {t_drop} s")
        print(f"  - 起爆延迟: {t_explode_delay} s")
        
        # 参数设置（与solve1.py一致）
        self.uav_speed = uav_speed
        self.flight_angle_offset = flight_angle_offset
        self.t_drop_after_task = t_drop
        self.t_explode_after_drop = t_explode_delay
        
        # 场景参数（与solve1.py完全一致）
        self.fake_target = np.array([0.0, 0.0, 0.0])
        self.real_target_center = np.array([0.0, 200.0, 0.0])
        self.real_target_radius = 7.0
        self.real_target_height = 10.0
        
        # 导弹参数（与solve1.py完全一致）
        self.missile_pos = np.array([20000.0, 0.0, 2000.0])
        self.missile_speed = 300.0
        self.missile_target = self.fake_target
        self.missile_distance = np.linalg.norm(self.missile_pos - self.missile_target)
        self.missile_direction = (self.missile_target - self.missile_pos) / self.missile_distance
        self.missile_flight_time = self.missile_distance / self.missile_speed
        
        # 无人机参数（与solve1.py完全一致）
        self.uav_pos = np.array([17800.0, 0.0, 1800.0])
        
        # 烟雾参数（与solve1.py完全一致）
        self.gravity = 9.8
        self.cloud_sink_speed = 3.0
        self.effective_radius = 10.0
        self.effective_duration = 20.0
        
        # 生成目标关键点（与solve1.py完全一致）
        self.target_key_points = self._generate_target_key_points()
        
        print(f"  - 导弹飞行总时间: {self.missile_flight_time:.2f} s")
        print(f"  - 目标关键点数量: {len(self.target_key_points)}")

    def _generate_target_key_points(self, num_points_per_circle=50):
        """生成真目标圆柱体上的关键点（与solve1.py完全一致）"""
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
        """计算点到线段的距离（与solve1.py完全一致）"""
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
        """计算无人机飞行方向（与solve1.py逻辑一致，但支持角度偏移）"""
        # 基准方向：朝向假目标
        uav_dir_vec_xy = self.fake_target[:2] - self.uav_pos[:2]
        base_direction_2d = uav_dir_vec_xy / np.linalg.norm(uav_dir_vec_xy)
        
        # 应用角度偏移
        cos_offset = np.cos(self.flight_angle_offset)
        sin_offset = np.sin(self.flight_angle_offset)
        
        rotated_dir_2d = np.array([
            cos_offset * base_direction_2d[0] - sin_offset * base_direction_2d[1],
            sin_offset * base_direction_2d[0] + cos_offset * base_direction_2d[1]
        ])
        
        uav_direction = np.zeros(3)
        uav_direction[:2] = rotated_dir_2d
        
        return uav_direction

    def calculate_blocking_duration(self):
        """计算遮蔽时长（与solve1.py完全一致的算法）"""
        # 1. 计算无人机飞行方向
        uav_direction = self.calculate_uav_direction()
        
        # 2. 计算烟雾弹投放位置（与solve1.py完全一致）
        uav_drop_pos = self.uav_pos + self.uav_speed * self.t_drop_after_task * uav_direction
        
        # 3. 计算烟雾弹起爆位置（与solve1.py完全一致）
        bomb_initial_velocity = self.uav_speed * uav_direction
        explode_pos = uav_drop_pos.copy()
        explode_pos += bomb_initial_velocity * self.t_explode_after_drop
        explode_pos[2] -= 0.5 * self.gravity * self.t_explode_after_drop**2
        
        t_explode_abs = self.t_drop_after_task + self.t_explode_after_drop

        if explode_pos[2] < 0:
            return 0.0, "烟雾弹在落地后起爆", {
                'uav_direction': uav_direction,
                'uav_drop_pos': uav_drop_pos,
                'explode_pos': explode_pos,
                't_explode_abs': t_explode_abs
            }

        # 4. 模拟遮蔽过程（与solve1.py完全一致）
        print("开始模拟遮蔽过程...")
        dt = 0.01  # 使用较大的时间步长以加快计算
        total_blocked_time = 0
        smoke_start_time = t_explode_abs
        smoke_end_time = smoke_start_time + self.effective_duration
        
        print(f"  - 起爆位置: [{explode_pos[0]:.1f}, {explode_pos[1]:.1f}, {explode_pos[2]:.1f}]")
        print(f"  - 模拟时间范围: {smoke_start_time:.3f} - {smoke_end_time:.3f} s")
        
        step_count = 0
        total_steps = int((min(smoke_end_time, self.missile_flight_time) - smoke_start_time) / dt)
        
        for t in np.arange(smoke_start_time, smoke_end_time, dt):
            if t > self.missile_flight_time:
                break
            
            current_missile_pos = self.missile_pos + self.missile_speed * t * self.missile_direction
            time_since_explode = t - smoke_start_time
            current_cloud_pos = explode_pos.copy()
            current_cloud_pos[2] -= self.cloud_sink_speed * time_since_explode
            
            if current_cloud_pos[2] < 0:
                break

            # 检查是否所有关键点都被遮蔽（与solve1.py完全一致）
            all_blocked = True
            blocked_count = 0
            for key_point in self.target_key_points:
                dist = self.point_to_line_distance(current_cloud_pos, current_missile_pos, key_point)
                if dist <= self.effective_radius:
                    blocked_count += 1
                else:
                    all_blocked = False
            
            if all_blocked:
                total_blocked_time += dt
            
            step_count += 1
            if step_count % 500 == 0:
                progress = step_count / total_steps * 100
                print(f"  - 进度: {progress:.1f}% (遮蔽点: {blocked_count}/{len(self.target_key_points)})")
        
        return total_blocked_time, "计算完成", {
            'uav_direction': uav_direction,
            'uav_drop_pos': uav_drop_pos,
            'explode_pos': explode_pos,
            't_explode_abs': t_explode_abs
        }

    def get_detailed_results(self):
        """获取详细的计算结果"""
        duration, status, details = self.calculate_blocking_duration()
        
        return {
            'blocking_duration': duration,
            'status': status,
            'explosion_time': details['t_explode_abs'],
            'explosion_pos': details['explode_pos'],
            'drop_pos': details['uav_drop_pos'],
            'uav_direction': details['uav_direction'],
            'flight_angle_deg': math.degrees(self.flight_angle_offset)
        }


def main():
    """主函数"""
    print("🎯 与solve1.py一致的参数化验证程序")
    print("="*60)
    
    # 使用solve1.py的默认参数
    problem = Solve1CompatibleInterference(
        uav_speed=128.455023,
        flight_angle_offset=-3.039914,
        t_drop=0.377478,
        t_explode_delay=0.491462
    )
    
    results = problem.get_detailed_results()
    
    print("\n" + "="*20 + " 验证结果 " + "="*20)
    print(f"✅ 有效遮蔽时长: {results['blocking_duration']:.6f} 秒")
    print(f"📊 计算状态: {results['status']}")
    print(f"⏰ 起爆时间: {results['explosion_time']:.3f} s")
    print(f"💥 起爆位置: [{results['explosion_pos'][0]:.1f}, {results['explosion_pos'][1]:.1f}, {results['explosion_pos'][2]:.1f}]")
    print(f"📍 投放位置: [{results['drop_pos'][0]:.1f}, {results['drop_pos'][1]:.1f}, {results['drop_pos'][2]:.1f}]")
    print(f"🧭 飞行角度偏移: {results['flight_angle_deg']:.2f}°")
    print("="*60)
    
    print("\n注意：此结果应该与solve1.py的结果完全一致！")


if __name__ == "__main__":
    main()