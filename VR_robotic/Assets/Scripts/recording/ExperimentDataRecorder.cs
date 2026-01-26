using System;
using System.IO;
using UnityEngine;

public class ExperimentDataRecorder : MonoBehaviour
{
    [Header("Participant Info")]
    public string participantID = "P01";

    [Header("Experiment Type (Choose One)")]
    public bool BrainWithDT;
    public bool BrainWithoutDT;
    public bool TumorWithDT;
    public bool TumorWithoutDT;

    [Header("Tracked Objects")]
    public Transform leftNeedleTip;
    public Transform rightNeedleTip;
    public Transform targetObject;

    [Header("Recording Options")]
    public float logInterval = 0.1f; // seconds between logs

    private float timer = 0f;
    private float startTime;
    private int collisionFrameCount = 0;  // 碰撞帧数
    private int collisionEventCount = 0;  // 碰撞次数（进入-退出算一次）
    private int frameCount = 0;
    private float fpsTime = 0f;

    private string experimentLabel;
    private StreamWriter writer;
    private string savePath;

    // Colliders
    private Collider leftCollider;
    private Collider rightCollider;
    private Collider targetCollider;

    // 状态机：上帧是否碰撞
    private bool wasCollidingLeft = false;
    private bool wasCollidingRight = false;

    void Start()
    {
        startTime = Time.time;

        // 确定实验类型标签
        if (BrainWithDT) experimentLabel = "Brain_With_DT";
        else if (BrainWithoutDT) experimentLabel = "Brain_Without_DT";
        else if (TumorWithDT) experimentLabel = "Tumor_With_DT";
        else if (TumorWithoutDT) experimentLabel = "Tumor_Without_DT";
        else experimentLabel = "Unknown";

        // 创建目录
        string folderPath = Path.Combine(Application.dataPath, "data", participantID);
        Directory.CreateDirectory(folderPath);

        // 文件名 = 实验类型 + 时间戳
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fileName = $"{experimentLabel}_{timestamp}.csv";
        savePath = Path.Combine(folderPath, fileName);

        writer = new StreamWriter(savePath);
        writer.WriteLine("timestamp,leftDistance,rightDistance,collisionFrames,collisionEvents,currentFPS");

        // 获取 Collider
        targetCollider = targetObject.GetComponent<Collider>();
        leftCollider = leftNeedleTip.GetComponent<Collider>();
        rightCollider = rightNeedleTip.GetComponent<Collider>();

        if (targetCollider == null || leftCollider == null || rightCollider == null)
        {
            Debug.LogError("Missing Collider on target or needle tips!");
        }
    }

    void Update()
    {
        timer += Time.deltaTime;
        fpsTime += Time.deltaTime;
        frameCount++;

        bool leftNow = leftCollider != null && targetCollider != null && leftCollider.bounds.Intersects(targetCollider.bounds);
        bool rightNow = rightCollider != null && targetCollider != null && rightCollider.bounds.Intersects(targetCollider.bounds);

        // 碰撞帧统计
        if (leftNow || rightNow)
            collisionFrameCount++;

        // 碰撞事件统计（进入时 +1）
        if (leftNow && !wasCollidingLeft) collisionEventCount++;
        if (rightNow && !wasCollidingRight) collisionEventCount++;

        // 更新状态
        wasCollidingLeft = leftNow;
        wasCollidingRight = rightNow;

        if (timer >= logInterval)
        {
            timer -= logInterval;

            // 使用 Collider.ClosestPoint 计算针尖到目标表面的真实距离
            float leftDist = Vector3.Distance(leftNeedleTip.position, targetCollider.ClosestPoint(leftNeedleTip.position));
            float rightDist = Vector3.Distance(rightNeedleTip.position, targetCollider.ClosestPoint(rightNeedleTip.position));

            float currentFPS = frameCount / fpsTime;

            writer.WriteLine($"{Time.time - startTime:F3},{leftDist:F4},{rightDist:F4},{collisionFrameCount},{collisionEventCount},{currentFPS:F2}");
            writer.Flush();

            frameCount = 0;
            fpsTime = 0f;
        }
    }
    public string GetExperimentLabel()
    {
        if (BrainWithDT) return "Brain_With_DT";
        else if (BrainWithoutDT) return "Brain_Without_DT";
        else if (TumorWithDT) return "Tumor_With_DT";
        else if (TumorWithoutDT) return "Tumor_Without_DT";
        else return "Unknown";
    }


    void OnApplicationQuit()
    {
        writer?.Close();
    }
}
