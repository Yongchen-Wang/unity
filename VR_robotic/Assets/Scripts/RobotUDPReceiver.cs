using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class RobotUDPReceiver : MonoBehaviour
{
    public List<GameObject> leftRobotFrontBack;
    public List<GameObject> leftRobotLeftRight;
    public List<GameObject> leftRobotUpDown;
    public List<GameObject> rightRobotFrontBack;
    public List<GameObject> rightRobotLeftRight;
    public List<GameObject> rightRobotUpDown;

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private Thread receiveThread;
    private bool isReceiving = true;

    private Queue<string> messageQueue = new Queue<string>();
    private object queueLock = new object();

    void Start()
    {
        // 设置 leftRobotUpDown 中每个物体的 Y 坐标为 0.1
        foreach (GameObject obj in leftRobotUpDown)
        {
            if (obj != null)
            {
                Vector3 pos = obj.transform.position;
                pos.y = 0.1f;
                obj.transform.position = pos;
            }
        }

        // 设置 rightRobotUpDown 中每个物体的 Y 坐标为 0.1
        foreach (GameObject obj in rightRobotUpDown)
        {
            if (obj != null)
            {
                Vector3 pos = obj.transform.position;
                pos.y = 0.1f;
                obj.transform.position = pos;
            }
        }

        udpClient = new UdpClient(5006);
        remoteEndPoint = new IPEndPoint(IPAddress.Any, 5006);

        // 启动后台线程接收 UDP
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
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
                Debug.LogWarning("UDP Receive Error: " + ex.Message);
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
    }

    void ProcessMessage(string message)
    {
        string[] parts = message.Split(',');
        if (parts.Length != 4) return;

        string identifier = parts[0]; // "L" or "R"
        if (!float.TryParse(parts[1], out float moveX)) return;
        if (!float.TryParse(parts[2], out float moveY)) return;
        if (!float.TryParse(parts[3], out float moveZ)) return;

        if (identifier == "L")
        {
            MoveRobots(leftRobotFrontBack, new Vector3(0, 0, -moveY));
            MoveRobots(leftRobotLeftRight, new Vector3(moveX, 0, 0));
            MoveRobots(leftRobotUpDown, new Vector3(0, -moveZ, 0));
        }
        else if (identifier == "R")
        {
            MoveRobots(rightRobotFrontBack, new Vector3(0, 0, moveY));
            MoveRobots(rightRobotLeftRight, new Vector3(-moveX, 0, 0));
            MoveRobots(rightRobotUpDown, new Vector3(0, -moveZ, 0));
        }
    }

    void MoveRobots(List<GameObject> robotParts, Vector3 displacement)
    {
        foreach (GameObject part in robotParts)
        {
            if (part != null)
            {
                part.transform.position += displacement;
            }
        }
    }

    void OnApplicationQuit()
    {
        isReceiving = false;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }

        udpClient.Close();
    }
}
