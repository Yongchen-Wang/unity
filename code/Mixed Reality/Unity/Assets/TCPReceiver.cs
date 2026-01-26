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
    private const int listenPort = 5005;

    async void Start()
    {
        tcpListener = new TcpListener(IPAddress.Any, listenPort);
        tcpListener.Start();
        Debug.Log("TCP Receiver started.");
        await StartAccepting();
    }

    async Task StartAccepting()
    {
        while (true)
        {
            TcpClient client = await tcpListener.AcceptTcpClientAsync();
            Debug.Log("Client connected.");
            HandleClient(client);
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
                while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) != 0)
                {
                    string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    ProcessReceivedData(message);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error with client connection: " + e.Message);
        }
        finally
        {
            client.Close();
        }
    }

    void ProcessReceivedData(string message)
    {
        // 使用 '\n' 分割每组数据
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
        tcpListener.Stop();
    }
}
