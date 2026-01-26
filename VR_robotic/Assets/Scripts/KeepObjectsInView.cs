using UnityEngine;

public class KeepObjectsInView : MonoBehaviour
{
    public Transform leftObject;  // 需要固定在左上角的物体
    public Transform rightObject; // 需要固定在右上角的物体
    public Camera mainCamera;
    public float offset = 0.1f; // 视口边缘的偏移量

    void Update()
    {
        if (mainCamera == null || leftObject == null || rightObject == null)
        {
            return;
        }

        // 计算左上角视口坐标并转换为世界坐标
        Vector3 leftViewportPosition = new Vector3(offset, 1 - offset, mainCamera.nearClipPlane + 1f);
        Vector3 leftWorldPosition = mainCamera.ViewportToWorldPoint(leftViewportPosition);
        leftObject.position = leftWorldPosition;

        // 计算右上角视口坐标并转换为世界坐标
        Vector3 rightViewportPosition = new Vector3(1 - offset, 1 - offset, mainCamera.nearClipPlane + 1f);
        Vector3 rightWorldPosition = mainCamera.ViewportToWorldPoint(rightViewportPosition);
        rightObject.position = rightWorldPosition;
    }
}
