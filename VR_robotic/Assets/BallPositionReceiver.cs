using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Threading;

/// <summary>
/// 小球位置接收器 - 高精度版本
/// 接收来自Python的TCP数据，将物理空间坐标转换为Unity模型坐标
/// 
/// 坐标系统说明:
/// - Python发送: (x_mm, y_mm, z_mm) 物理空间坐标（毫米）
/// - Unity使用: (x, y, z) 模型局部坐标（Unity单位）
/// - 映射关系: Python的XY平面对应Unity的XY平面（正视角度）
/// </summary>
public class BallPositionReceiver : MonoBehaviour
{
    [Header("坐标转换参数")]
    [Tooltip("实际血管模型尺寸（毫米）")]
    public float realVesselSize_mm = 18f;  // 实际血管模型：18mm x 18mm
    
    [Tooltip("Unity模型尺寸（Unity单位）")]
    public float unityVesselSize = 18f;  // Unity模型：15 x 15 Unity单位
    
    [Tooltip("固定的Z坐标（深度）\n小球在XY平面上移动，Z轴保持固定")]
    public float fixedZ = -2.5f;  // ⭐ 小球固定在Z=-3平面
    
    [Tooltip("X轴是否反转（根据坐标系对应关系调整）")]
    public bool invertX = false;
    
    [Tooltip("Y轴是否反转（根据坐标系对应关系调整）")]
    public bool invertY = false;
    
    [Header("调试选项")]
    [Tooltip("是否显示调试信息")]
    public bool showDebugInfo = false;
    
    [Tooltip("调试信息更新间隔（秒）")]
    public float debugInterval = 0.5f;
    
    private Vector3 startPosition;
    private float lastDebugTime = 0f;

    void Start()
    {
        startPosition = transform.localPosition;
        
        Debug.Log("=== BallPositionReceiver 初始化 ===");
        Debug.Log($"坐标映射: Python XY平面 -> Unity XY平面");
        Debug.Log($"实际血管尺寸: {realVesselSize_mm} mm");
        Debug.Log($"Unity模型尺寸: {unityVesselSize} Unity单位");
        Debug.Log($"固定Z轴: {fixedZ}");
        Debug.Log($"X轴反转: {invertX}, Y轴反转: {invertY}");
        Debug.Log($"初始位置: {startPosition}");
        Debug.Log("====================================");
    }

    void Update()
    {
        // 从TCPReceiver获取最新位置
        // TCPReceiver.latestPosition 格式: (x_mm, y_mm, z_mm)
        Vector3 receivedPos = TCPReceiver.latestPosition;
        
        // ⭐ 坐标转换: Python XY平面 -> Unity XY平面
        // Python坐标系统: 原点在左下角，X向右，Y向上 (0-18mm)
        // Unity坐标系统: 原点在左上角，X向右，Y向下 (0-15 Unity单位)
        // 
        // 映射公式: 将Python的0-18mm映射到Unity的0-15单位
        // X: (Python X / 18mm) * 15 Unity单位
        // Y: (Python Y / 18mm) * 15 Unity单位 (同向映射)
        
        float newX = (receivedPos.x / realVesselSize_mm) * unityVesselSize;
        if (invertX) newX = -newX;
        
        // Y轴映射：与Python Y同向（不使用翻转）
        float newY = (receivedPos.y / realVesselSize_mm) * unityVesselSize;
        if (invertY) newY = -newY;
        
        float newZ = fixedZ;  // ⭐ Z轴固定（深度方向）
        
        // 设置新位置
        transform.localPosition = new Vector3(newX, newY, newZ);
        
        // 调试信息（每0.5秒打印一次）
        if (showDebugInfo && Time.time - lastDebugTime > debugInterval)
        {
            Debug.Log($"[Ball] 接收: ({receivedPos.x:F2}, {receivedPos.y:F2}, {receivedPos.z:F2}) mm " +
                     $"-> Unity: ({newX:F4}, {newY:F4}, {newZ:F4})");
            lastDebugTime = Time.time;
        }
    }
    
    /// <summary>
    /// 根据真实测量调整血管尺寸参数
    /// 使用方法：
    /// 1. 在Python端标定区域放置小球在已知位置（如0,0）
    /// 2. 在Unity中测量小球模型的实际位置
    /// 3. 调整realVesselSize_mm和unityVesselSize直到位置匹配
    /// </summary>
    public void CalibrateVesselSize(float realSize_mm, float unitySize)
    {
        realVesselSize_mm = realSize_mm;
        unityVesselSize = unitySize;
        Debug.Log($"标定完成！实际尺寸: {realVesselSize_mm} mm, Unity尺寸: {unityVesselSize}");
    }
}
