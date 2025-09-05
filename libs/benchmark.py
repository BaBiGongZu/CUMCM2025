
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
            print(f"\n测试 {n:,} 次迭代...")
            elapsed = c_lib.performance_test(n)
            ops_per_sec = n / elapsed if elapsed > 0 else float('inf')
            print(f"  耗时: {elapsed:.4f} 秒")
            print(f"  性能: {ops_per_sec:,.0f} 操作/秒")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    benchmark_performance()
