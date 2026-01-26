using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class TCPSend : MonoBehaviour, IMixedRealityHandJointHandler
{
    private Vector3 previousIndexTipPosition;
    private bool hasPreviousIndexTipPosition = false;
    private Vector3 cumulativeDelta; // 累积增量
    private float timeSinceLastSend = 0.0f; // 自上次发送以来的时间
    private const float sendInterval = 0.1f; // 发送间隔为0.1秒

    private TcpClient tcpClient;
    private NetworkStream stream;
    private string targetIP = "127.0.0.1"; // 电脑的局域网IP地址
    //private string targetIP = "192.168.137.1"; // 电脑的局域网IP地址
    private int targetPort = 12345; // 与Python脚本端口匹配

    public Vector3 IndexTipPositionDelta { get; private set; }
    public Vector3 CumulativeDelta // 公共属性来获取累积增量
    {
        get { return cumulativeDelta; }
    }

    void Start()
    {
        tcpClient = new TcpClient();
        tcpClient.Connect(targetIP, targetPort); // 连接到服务器
        stream = tcpClient.GetStream(); // 获取用于发送数据的网络流
    }

    void Update()
    {
        timeSinceLastSend += Time.deltaTime; // 更新时间
        if (timeSinceLastSend >= sendInterval)
        {
            // 是时候发送数据了
            if (cumulativeDelta != Vector3.zero) // 仅当有数据时发送
            {
                SendIndexTipPosition(CumulativeDelta);
                cumulativeDelta = Vector3.zero; // 重置累积增量
            }
            timeSinceLastSend = 0.0f; // 重置计时器
        }
    }


    public void OnHandJointsUpdated(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData)
    {
        if (eventData.Handedness == Handedness.Left) // 这里从Right更改为Left
        {
            if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexTip, out MixedRealityPose indexTipPose))
            {
                if (hasPreviousIndexTipPosition)
                {
                    IndexTipPositionDelta = indexTipPose.Position - previousIndexTipPosition;
                    cumulativeDelta += IndexTipPositionDelta;
                }
                previousIndexTipPosition = indexTipPose.Position;
                hasPreviousIndexTipPosition = true;
            }
        }
    }

    private void SendIndexTipPosition(Vector3 position)
    {
        // 将Vector3转换为字符串
        string message = $"{position.x},{position.y},{position.z}";
        byte[] sendBytes = Encoding.UTF8.GetBytes(message);
        stream.Write(sendBytes, 0, sendBytes.Length); // 通过网络流发送数据
    }

    void OnEnable()
    {
        CoreServices.InputSystem?.RegisterHandler<IMixedRealityHandJointHandler>(this);
    }

    void OnDisable()
    {
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealityHandJointHandler>(this);
        hasPreviousIndexTipPosition = false; // 重置标志
        cumulativeDelta = Vector3.zero; // 重置累积增量

        if (tcpClient != null)
        {
            stream.Close();
            tcpClient.Close();
            tcpClient = null;
        }
    }

    void OnDestroy()
    {
        if (tcpClient != null)
        {
            stream.Close();
            tcpClient.Close();
        }
    }
}
