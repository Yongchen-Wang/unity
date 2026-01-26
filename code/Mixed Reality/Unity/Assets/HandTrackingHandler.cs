using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class HandTrackingHandler : MonoBehaviour, IMixedRealityHandJointHandler
{
    private Vector3 previousIndexTipPosition;
    private bool hasPreviousIndexTipPosition = false;
    private Vector3 cumulativeDelta; // 累积增量

    private UdpClient udpClient;
    private string targetIP = "192.168.137.1"; // 电脑的局域网IP地址
    private int targetPort = 12345; // 与Python脚本端口匹配

    public Vector3 IndexTipPositionDelta { get; private set; }
    public Vector3 CumulativeDelta // 公共属性来获取累积增量
    {
        get { return cumulativeDelta; }
    }

    void Start()
    {
        udpClient = new UdpClient();
        udpClient.Connect(targetIP, targetPort);
    }

    public void OnHandJointsUpdated(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData)
    {
        if (eventData.Handedness == Handedness.Right)
        {
            if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexTip, out MixedRealityPose indexTipPose))
            {
                if (hasPreviousIndexTipPosition)
                {
                    IndexTipPositionDelta = indexTipPose.Position - previousIndexTipPosition;
                    cumulativeDelta += IndexTipPositionDelta;
                    // 将累积增量发送出去
                    SendIndexTipPosition(CumulativeDelta);
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
        udpClient.Send(sendBytes, sendBytes.Length);
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

        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}