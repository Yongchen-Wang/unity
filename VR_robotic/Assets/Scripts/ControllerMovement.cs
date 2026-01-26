using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ControllerBasedMovement : MonoBehaviour
{
    public GameObject targetObject;          // 被控制的 GameObject，例如 XR Origin
    public float moveSpeed = 1.5f;           // 移动速度
    public float rotationSpeed = 60f;        // 旋转速度

    void Update()
    {
        float step = Time.deltaTime;

        if (targetObject == null)
            return;

        // === 左手手柄控制位置移动 ===
        Vector2 leftStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);  // 左手柄摇杆

        // 获取方向（按对象当前朝向）
        Vector3 forward = targetObject.transform.forward;
        Vector3 right = targetObject.transform.right;
        forward.y = 0;
        right.y = 0;
        forward.Normalize();
        right.Normalize();

        Vector3 moveDir = (forward * leftStick.y + right * leftStick.x) * moveSpeed * step;
        targetObject.transform.position += moveDir;

        // === 右手手柄控制水平旋转 ===
        Vector2 rightStick = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);  // 右手柄摇杆
        float yaw = rightStick.x * rotationSpeed * step;

        // 只做 Y 轴旋转
        targetObject.transform.Rotate(Vector3.up, yaw, Space.World);
    }
}
