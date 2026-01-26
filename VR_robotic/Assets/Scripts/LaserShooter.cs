using UnityEngine;

public class LaserShooter : MonoBehaviour
{
    public float laserLength = 10f; // 激光长度
    public Color laserColor = Color.red; // 激光颜色
    private LineRenderer lineRenderer;

    void Start()
    {
        // 添加 LineRenderer 组件
        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = new Material(Shader.Find("Unlit/Color"));
        lineRenderer.material.color = laserColor;
        lineRenderer.positionCount = 2;
    }

    void Update()
    {
        // 计算激光起点和终点
        Vector3 startPosition = transform.position;
        Vector3 endPosition = startPosition + transform.right * laserLength;

        // 设置 LineRenderer 位置
        lineRenderer.SetPosition(0, startPosition);
        lineRenderer.SetPosition(1, endPosition);
    }
}
