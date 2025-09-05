import subprocess
import sys
import os
import platform
import shutil

def detect_platform():
    """检测当前平台"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == 'darwin':
        if machine in ['arm64', 'aarch64']:
            return 'macos_arm64'
        else:
            return 'macos_x64'
    elif system == 'linux':
        if machine in ['x86_64', 'amd64']:
            return 'linux_x64'
        elif machine in ['arm64', 'aarch64']:
            return 'linux_arm64'
    elif system == 'windows':
        return 'windows_x64'
    
    return 'unknown'

def build_library(platform_target=None):
    """构建动态库"""
    if platform_target is None:
        platform_target = detect_platform()
    
    print(f"构建平台: {platform_target}")
    
    # 创建输出目录
    lib_dir = "."
    
    source_file = "smoke_blocking.c"
    
    if platform_target.startswith('macos'):
        # macOS - 使用clang
        output_file = f"{lib_dir}/smoke_blocking_macos.dylib"
        cmd = [
            'clang',
            '-shared',
            '-fPIC',
            '-O3',
            '-march=native',  # 针对当前CPU优化
            '-ffast-math',    # 快速数学运算
            '-funroll-loops', # 循环展开
            '-o', output_file,
            source_file,
            '-lm'
        ]
        
        # 如果是Apple Silicon，添加特定优化
        if platform_target == 'macos_arm64':
            cmd.extend(['-mcpu=apple-m4'])
            
    elif platform_target.startswith('linux'):
        # Linux - 使用gcc
        output_file = f"{lib_dir}/smoke_blocking_linux.so"
        cmd = [
            'gcc',
            '-shared',
            '-fPIC',
            '-O3',
            '-march=native',
            '-ffast-math',
            '-funroll-loops',
            '-o', output_file,
            source_file,
            '-lm'
        ]
        
    elif platform_target == 'windows_x64':
        # Windows - 使用MinGW或MSVC
        output_file = f"{lib_dir}/smoke_blocking_windows.dll"
        # 优先尝试clang-cl (LLVM)

        cmd = [
            'gcc',
            '-shared',
            '-O3',
            '-o', output_file,
            source_file,
            '-lm'
        ]
    
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 构建成功: {output_file}")
            return output_file
        else:
            print(f"❌ 构建失败:")
            print(f"stderr: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return None
            
    except FileNotFoundError as e:
        print(f"❌ 编译器未找到: {e}")
        return None

def create_benchmark():
    """创建性能测试脚本"""
    benchmark_code = '''
# filepath: /Users/penner/Desktop/CUMCM2025/benchmark.py
import ctypes
import time
import os
import platform

def load_library():
    """根据平台加载对应的动态库"""
    system = platform.system().lower()
    
    if system == 'darwin':
        lib_path = 'smoke_blocking_macos.dylib'
    elif system == 'linux':
        lib_path = 'smoke_blocking_linux.so'
    elif system == 'windows':
        lib_path = 'smoke_blocking_windows.dll'
    else:
        raise RuntimeError(f"不支持的平台: {system}")
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"动态库不存在: {lib_path}")
    
    return ctypes.CDLL(lib_path)

def benchmark_performance():
    """性能基准测试"""
    print("🚀 C库性能基准测试")
    print("=" * 50)
    
    try:
        c_lib = load_library()
        
        # 设置函数签名
        c_lib.performance_test.argtypes = [ctypes.c_int]
        c_lib.performance_test.restype = ctypes.c_double
        
        c_lib.get_version.restype = ctypes.c_char_p
        
        # 获取版本信息
        version = c_lib.get_version().decode('utf-8')
        print(f"库版本: {version}")
        print(f"平台: {platform.system()} {platform.machine()}")
        
        # 性能测试
        iterations = [1000, 10000, 100000]
        
        for n in iterations:
            print(f"\\n测试 {n:,} 次迭代...")
            elapsed = c_lib.performance_test(n)
            ops_per_sec = n / elapsed if elapsed > 0 else float('inf')
            print(f"  耗时: {elapsed:.4f} 秒")
            print(f"  性能: {ops_per_sec:,.0f} 操作/秒")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    benchmark_performance()
'''
    
    with open('benchmark.py', 'w') as f:
        f.write(benchmark_code)
    
    print("✅ 创建性能测试脚本: benchmark.py")

def main():
    """主函数"""
    print("🔨 跨平台C库构建脚本")
    print("=" * 50)
    
    current_platform = detect_platform()
    print(f"检测到平台: {current_platform}")
    
    # 构建当前平台的库
    result = build_library(current_platform)
    
    if result:
        print(f"\n✅ 构建完成!")
        print(f"动态库位置: {result}")
        
        # 创建性能测试脚本
        create_benchmark()
        
        print(f"\n📋 使用说明:")
        print(f"1. 运行性能测试: python benchmark.py")
        print(f"2. 在其他平台上运行此脚本以构建对应的库")
        print(f"3. 所有库都会保存在 libs/ 目录中")
        
    else:
        print(f"\n❌ 构建失败!")
        
        # 提供故障排除建议
        print(f"\n🔧 故障排除:")
        if current_platform.startswith('macos'):
            print(f"- 确保安装了Xcode Command Line Tools: xcode-select --install")
        elif current_platform.startswith('linux'):
            print(f"- 确保安装了GCC: sudo apt-get install build-essential")
        elif current_platform == 'windows_x64':
            print(f"- 安装Visual Studio Build Tools或MinGW")

if __name__ == "__main__":
    main()