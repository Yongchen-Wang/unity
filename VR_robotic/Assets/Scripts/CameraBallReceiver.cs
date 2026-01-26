using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections.Generic;

public class CameraBallReceiver : MonoBehaviour
{
    [Header("UDP 设置")]
    public int receivePort = 5007;  // 接收小球位置的端口

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private Thread receiveThread;
    private bool isReceiving = true;

    [Header("摄像头坐标范围（和 Python 中 space1/space2 对应）")]
    public Vector2 spaceMin = new Vector2(-0.125f, 0.875f);  // 对应 camera_final.py 里的 space1
    public Vector2 spaceMax = new Vector2(0f, 1f);           // 对应 camera_final.py 里的 space2

    [Header("小球在血管中的运动范围（Unity 坐标）")]
    public Vector3 sceneMin = new Vector3(-0.05f, 0.0f, -0.05f);
    public Vector3 sceneMax = new Vector3(0.05f, 0.0f, 0.05f);

    private Queue<string> messageQueue = new Queue<string>();
    private object queueLock = new object();

    private Vector2 latestPos;   // 最新 (x,y)
    private bool hasPos = false;

    void Start()
    {
        try
        {
            udpClient = new UdpClient(receivePort);
            remoteEndPoint = new IPEndPoint(IPAddress.Any, receivePort);

            // 启动后台线程接收 UDP
            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log($"Listening for ball position on UDP port {receivePort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to start UDP receiver: {e.Message}");
        }
    }

    void ReceiveData()
    {
        while (isReceiving)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);

                lock (queueLock)
                {
                    messageQueue.Enqueue(message);
                }
            }
            catch (Exception ex)
            {
                if (isReceiving)
                {
                    Debug.LogWarning("UDP Receive Error: " + ex.Message);
                }
            }
        }
    }

    void Update()
    {
        // 在主线程中处理接收到的消息
        lock (queueLock)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                ProcessMessage(message);
            }
        }

        if (!hasPos) return;

        // 把 Python 归一化空间 (spaceMin, spaceMax) 映射到 Unity 中的小球运动范围 (sceneMin, sceneMax)
        float tx = Mathf.InverseLerp(spaceMin.x, spaceMax.x, latestPos.x);
        float ty = Mathf.InverseLerp(spaceMin.y, spaceMax.y, latestPos.y);

        // 这里示例：用 tx 控制 X，ty 控制 Z，Y 固定（球在"血管截面"中移动）
        float newX = Mathf.Lerp(sceneMin.x, sceneMax.x, tx);
        float newZ = Mathf.Lerp(sceneMin.z, sceneMax.z, ty);
        float newY = Mathf.Lerp(sceneMin.y, sceneMax.y, 0.5f); // 如果想固定高度，也可以直接写常数

        transform.localPosition = new Vector3(newX, newY, newZ);
    }

    void ProcessMessage(string message)
    {
        // 收到 "x,y"
        var parts = message.Split(',');
        if (parts.Length == 2 &&
            float.TryParse(parts[0], out float x) &&
            float.TryParse(parts[1], out float y))
        {
            latestPos = new Vector2(x, y);
            hasPos = true;
        }
    }

    void OnApplicationQuit()
    {
        isReceiving = false;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }

        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}

