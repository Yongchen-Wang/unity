using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Events;

public class HapticUDP : MonoBehaviour
{
    public HapticPlugin HPlugin = null;  // 绑定 Haptic 设备
    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    public UnityEvent OnHoldButton2;  // 按住按钮事件
    public UnityEvent OnReleaseButton2;  // 松开按钮事件

    private Vector3 lastPosition; // 记录上次松开时的位置
    private bool isHolding = false;  // 是否正在按住按钮

    void Start()
    {
        udpClient = new UdpClient();
        remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5005); // 目标 IP & 端口

        // 绑定按钮事件
        OnHoldButton2.AddListener(StartSending);
        OnReleaseButton2.AddListener(StopSending);
    }

    void StartSending()
    {
        if (!isHolding)
        {
            isHolding = true;
            lastPosition = HPlugin.CurrentPosition; // **按下按钮时，更新起始位置**
            StartCoroutine(SendPosition());
        }
    }

    public void StopSending()
    {
        isHolding = false;
        lastPosition = HPlugin.CurrentPosition; // **松开按钮时，保存当前位置**
    }

    IEnumerator SendPosition()
    {
        while (isHolding)
        {
            Vector3 currentPosition = HPlugin.CurrentPosition;
            print(currentPosition);
            Vector3 displacement = currentPosition - lastPosition; // 计算相对位移
            lastPosition = currentPosition; // **更新 lastPosition**

            // 直接发送原始位移（Unity 单位）
            string message = $"{displacement.x},{displacement.y},{displacement.z}";
            byte[] data = Encoding.UTF8.GetBytes(message);
            udpClient.Send(data, data.Length, remoteEndPoint);

            yield return new WaitForSeconds(0.1f); // 每 0.1s 发送一次
        }
    }

    void OnApplicationQuit()
    {
        udpClient.Close();
    }
}
