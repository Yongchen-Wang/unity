using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class TCPReceiver : MonoBehaviour
{
    public static Vector3 latestPosition;
    private TcpListener tcpListener;
    private const int listenPort = 5006;  // ⭐ 使用5006端口，由robot_data_logger.py转发数据
    private bool isRunning = false;

    async void Start()
    {
        try
        {
            isRunning = true;
            tcpListener = new TcpListener(IPAddress.Any, listenPort);
            tcpListener.Start();
            Debug.Log("TCP Receiver started on port " + listenPort);
            await StartAccepting();
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to start TCP Receiver: " + e.Message);
        }
    }

    async Task StartAccepting()
    {
        while (isRunning) // ⭐ 检查运行标志
        {
            try
            {
                TcpClient client = await tcpListener.AcceptTcpClientAsync();
                Debug.Log("Client connected.");
                HandleClient(client);
            }
            catch (ObjectDisposedException)
            {
                // TcpListener已被停止，正常退出
                Debug.Log("TCP Listener stopped.");
                break;
            }
            catch (Exception e)
            {
                if (isRunning) // 只在运行时记录错误
                {
                    Debug.LogError("Error accepting client: " + e.Message);
                }
                break;
            }
        }
    }

    async void HandleClient(TcpClient client)
    {
        try
        {
            using (NetworkStream stream = client.GetStream())
            {
                byte[] buffer = new byte[client.ReceiveBufferSize];
                int bytesRead;
                while (isRunning && (bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) != 0) // ⭐ 检查isRunning
                {
                    string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    ProcessReceivedData(message);
                }
            }
        }
        catch (ObjectDisposedException)
        {
            // Socket已关闭，正常退出
            Debug.Log("Client connection closed.");
        }
        catch (Exception e)
        {
            if (isRunning) // 只在运行时记录错误
            {
                Debug.LogError("Error with client connection: " + e.Message);
            }
        }
        finally
        {
            try
            {
                client?.Close();
            }
            catch { /* 忽略关闭时的异常 */ }
        }
    }

    void ProcessReceivedData(string message)
    {
        // ʹ�� '\n' �ָ�ÿ������
        string[] messages = message.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var msg in messages)
        {
            //Debug.Log("Received raw message: " + msg);

            string[] parts = msg.Split(',');
            if (parts.Length == 3)
            {
                try
                {
                    float x = float.Parse(parts[0], CultureInfo.InvariantCulture);
                    float y = float.Parse(parts[1], CultureInfo.InvariantCulture);
                    float z = float.Parse(parts[2], CultureInfo.InvariantCulture);
                    latestPosition = new Vector3(x, y, z);
                    //Debug.Log($"Received position: {latestPosition}");
                }
                catch (FormatException e)
                {
                    Debug.LogError("Received non-numeric data: " + e.Message);
                }
            }
            else
            {
                Debug.LogError("Received data in unexpected format: " + msg);
            }
        }
    }

    private void OnDestroy()
    {
        // ⭐ 正确清理TCP连接
        isRunning = false; // 停止接受新连接
        
        if (tcpListener != null)
        {
            try
            {
                tcpListener.Stop();
                Debug.Log("TCP Receiver stopped.");
            }
            catch (Exception e)
            {
                Debug.LogWarning("Error stopping TCP Receiver: " + e.Message);
            }
        }
    }
    
    // ⭐ 添加应用程序退出时的清理
    private void OnApplicationQuit()
    {
        OnDestroy();
    }
}
