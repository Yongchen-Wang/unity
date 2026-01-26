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
/// - 映射关系: Python的XY平面对应Unity的XZ平面（俯视角度）
/// </summary>
public class BallPositionReceiver : MonoBehaviour
{
    [Header("坐标转换参数")]
    [Tooltip("缩放因子：物理毫米到Unity单位的转换\n例如：如果125mm对应Unity的1.25单位，scale=100")]
    public float scale = 100000f;
    
    [Tooltip("固定的Y坐标（高度）\n小球在平面上移动，Y轴保持固定")]
    public float fixedY = 0.03f;
    
    [Tooltip("X轴是否反转（根据坐标系对应关系调整）")]
    public bool invertX = true;
    
    [Tooltip("Z轴是否反转（根据坐标系对应关系调整）")]
    public bool invertZ = false;
    
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
        Debug.Log($"缩放因子: {scale}");
        Debug.Log($"固定Y轴: {fixedY}");
        Debug.Log($"X轴反转: {invertX}, Z轴反转: {invertZ}");
        Debug.Log($"初始位置: {startPosition}");
        Debug.Log("====================================");
    }

    void Update()
    {
        // 从TCPReceiver获取最新位置
        // TCPReceiver.latestPosition 格式: (x_mm, y_mm, z_mm)
        Vector3 receivedPos = TCPReceiver.latestPosition;
        
        // 坐标转换:
        // Python X (横向) -> Unity X (或 -X)
        // Python Y (纵向) -> Unity Z (或 -Z)  
        // Python Z (高度) -> Unity Y (通常固定)
        
        float newX = receivedPos.x / scale;
        if (invertX) newX = -newX;
        
        float newY = fixedY; // 小球在平面上，Y轴固定
        
        float newZ = receivedPos.y / scale;
        if (invertZ) newZ = -newZ;
        
        // 设置新位置
        transform.localPosition = new Vector3(newX, newY, newZ);
        
        // 调试信息
        if (showDebugInfo && Time.time - lastDebugTime > debugInterval)
        {
            Debug.Log($"[Ball] 接收: ({receivedPos.x:F2}, {receivedPos.y:F2}, {receivedPos.z:F2}) mm " +
                     $"-> Unity: ({newX:F4}, {newY:F4}, {newZ:F4})");
            lastDebugTime = Time.time;
        }
    }
    
    /// <summary>
    /// 根据真实测量调整缩放因子
    /// 使用方法：
    /// 1. 在Python端标定区域放置小球在已知位置（如0,0）
    /// 2. 在Unity中测量小球模型的实际位置
    /// 3. 调整scale直到位置匹配
    /// </summary>
    public void CalibrateScale(float realDistance_mm, float unityDistance)
    {
        scale = realDistance_mm / unityDistance;
        Debug.Log($"标定完成！新的缩放因子: {scale}");
        Debug.Log($"物理距离 {realDistance_mm} mm = Unity距离 {unityDistance}");
    }
}
