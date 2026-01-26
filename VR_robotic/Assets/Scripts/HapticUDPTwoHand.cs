using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using System.IO;
using System;

public class HapticUDPTwoHand : MonoBehaviour
{
    public HapticPlugin HPluginL = null;  // �����豸
    public HapticPlugin HPluginR = null;  // �����豸

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    private Vector3 lastPositionL; // �����ϴ�λ��
    private Vector3 lastPositionR; // �����ϴ�λ��

    public bool isMovingR = false; // �����Ƿ����ڷ�������
    public bool isMovingL = false; // �����Ƿ����ڷ�������

    private bool isLogging = false;
    private const string LoggerHost = "127.0.0.1";
    private const int LoggerPort = 5008;
    private const string DebugLogPath = @"d:\project\individual_project\unity\.cursor\debug.log";

    void Start()
    {
        udpClient = new UdpClient();
        remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5005); // Ŀ�� IP & �˿�

        // ��ʼ�� lastPosition
        lastPositionL = HPluginL.CurrentPosition;
        lastPositionR = HPluginR.CurrentPosition;

        isMovingL = false;
        isMovingR = false;

        // #region agent log
        DebugLog(
            "HapticUDPTwoHand.cs:Start",
            "start",
            $"{{\"hasHPluginL\":{(HPluginL != null).ToString().ToLower()},\"hasHPluginR\":{(HPluginR != null).ToString().ToLower()}}}"
        );
        // #endregion
    }
    private void Awake()
    {
        isMovingL = false;
        isMovingR = false;
    }

    public void ToggleRightHand()
    {
        isMovingR = !isMovingR;  // �л�״̬

        if (isMovingR)
        {
            lastPositionR = HPluginR.CurrentPosition; // ��¼��ǰλ��
            StartCoroutine(SendRightHandData()); // ��ʼ��������
            Debug.Log("Right hand movement started.");
        }
        else
        {
            StopCoroutine(SendRightHandData()); // ֹͣ��������
            Debug.Log("Right hand movement stopped.");
        }

        UpdateLoggerState();
    }

    public void ToggleLeftHand()
    {
        isMovingL = !isMovingL;  // �л�״̬

        if (isMovingL)
        {
            lastPositionL = HPluginL.CurrentPosition; // ��¼��ǰλ��
            StartCoroutine(SendLeftHandData()); // ��ʼ��������
            Debug.Log("Left hand movement started.");
        }
        else
        {
            StopCoroutine(SendLeftHandData()); // ֹͣ��������
            Debug.Log("Left hand movement stopped.");
        }

        UpdateLoggerState();
    }

    private void UpdateLoggerState()
    {
        bool shouldLog = isMovingL || isMovingR;
        if (shouldLog && !isLogging)
        {
            // #region agent log
            DebugLog(
                "HapticUDPTwoHand.cs:UpdateLoggerState:START",
                "logger start requested",
                $"{{\"isMovingL\":{isMovingL.ToString().ToLower()},\"isMovingR\":{isMovingR.ToString().ToLower()}}}"
            );
            // #endregion
            SendLoggerCommand("START");
            isLogging = true;
        }
        else if (!shouldLog && isLogging)
        {
            // #region agent log
            DebugLog(
                "HapticUDPTwoHand.cs:UpdateLoggerState:STOP",
                "logger stop requested",
                $"{{\"isMovingL\":{isMovingL.ToString().ToLower()},\"isMovingR\":{isMovingR.ToString().ToLower()}}}"
            );
            // #endregion
            SendLoggerCommand("STOP");
            isLogging = false;
        }
    }

    private void SendLoggerCommand(string command)
    {
        try
        {
            // #region agent log
            DebugLog(
                "HapticUDPTwoHand.cs:SendLoggerCommand:attempt",
                "logger command attempt",
                $"{{\"command\":\"{command}\",\"host\":\"{LoggerHost}\",\"port\":{LoggerPort}}}"
            );
            // #endregion
            using (TcpClient client = new TcpClient())
            {
                client.Connect(LoggerHost, LoggerPort);
                byte[] data = Encoding.UTF8.GetBytes(command);
                NetworkStream stream = client.GetStream();
                stream.Write(data, 0, data.Length);
            }
            // #region agent log
            DebugLog(
                "HapticUDPTwoHand.cs:SendLoggerCommand:success",
                "logger command sent",
                $"{{\"command\":\"{command}\"}}"
            );
            // #endregion
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Logger command failed: {ex.Message}");
            // #region agent log
            DebugLog(
                "HapticUDPTwoHand.cs:SendLoggerCommand:error",
                "logger command failed",
                $"{{\"command\":\"{command}\",\"error\":\"{ex.Message.Replace("\"", "'")}\"}}"
            );
            // #endregion
        }
    }

    private void DebugLog(string location, string message, string dataJson)
    {
        try
        {
            string dir = Path.GetDirectoryName(DebugLogPath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }
            string entry =
                $"{{\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"H11\",\"location\":\"{location}\",\"message\":\"{message}\",\"data\":{dataJson},\"timestamp\":{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}}}\n";
            File.AppendAllText(DebugLogPath, entry);
        }
        catch
        {
            // ignore logging errors
        }
    }

    IEnumerator SendRightHandData()
    {
        while (isMovingR)
        {
            Vector3 currentPositionR = HPluginR.CurrentPosition;
            Vector3 displacementR = currentPositionR - lastPositionR;
            lastPositionR = currentPositionR;

            // ���� UDP ���ݣ����֣�
            string messageR = $"R,{displacementR.x},{displacementR.y},{displacementR.z}";
            byte[] dataR = Encoding.UTF8.GetBytes(messageR);
            udpClient.Send(dataR, dataR.Length, remoteEndPoint);
            // �ڷ��ͺ�׷�ӣ�
            //LogMovementToCSV("R", displacementR);


            yield return new WaitForSeconds(0.1f); // ÿ 0.1s ����һ��
        }
    }

    IEnumerator SendLeftHandData()
    {
        while (isMovingL)
        {
            Vector3 currentPositionL = HPluginL.CurrentPosition;
            Vector3 displacementL = currentPositionL - lastPositionL;
            lastPositionL = currentPositionL;

            // ���� UDP ���ݣ����֣�
            string messageL = $"L,{displacementL.x},{displacementL.y},{displacementL.z}";
            byte[] dataL = Encoding.UTF8.GetBytes(messageL);
            udpClient.Send(dataL, dataL.Length, remoteEndPoint);
            // �ڷ��ͺ�׷�ӣ�
            //LogMovementToCSV("L", displacementL);


            yield return new WaitForSeconds(0.1f); // ÿ 0.1s ����һ��

        }
    }

    void OnApplicationQuit()
    {
        if (isLogging)
        {
            SendLoggerCommand("STOP");
            isLogging = false;
        }
        udpClient.Close();
    }

    private void LogMovementToCSV(string hand, Vector3 displacement)
    {
        string id = "UnknownID";
        string experimentLabel = "Unknown";
        string folderPath = "";

        // ��ȡ������ ID ��ʵ������
        ExperimentDataRecorder recorder = FindObjectOfType<ExperimentDataRecorder>();
        if (recorder != null)
        {
            id = recorder.participantID;
            experimentLabel = recorder.GetExperimentLabel(); // ����һ����������ȡʵ�������ַ���
        }

        folderPath = Path.Combine(Application.dataPath, "data", id);
        Directory.CreateDirectory(folderPath);

        // �ļ�����ʵ������ + ʱ��� + move_input.csv
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fileName = $"{experimentLabel}_{timestamp}_move_input.csv";
        string filePath = Path.Combine(folderPath, fileName);

        // ����ļ������ڣ�д���ͷ
        if (!File.Exists(filePath))
        {
            File.WriteAllText(filePath, "timestamp,hand,dx,dy,dz\n");
        }

        float timestampNow = Time.time;
        string line = $"{timestampNow:F3},{hand},{displacement.x},{displacement.y},{displacement.z}";
        File.AppendAllText(filePath, line + "\n");
    }


}