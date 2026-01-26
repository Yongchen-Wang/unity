using Unity.VisualScripting;
using UnityEngine;

public class RobotMovement : MonoBehaviour
{
    public float moveSpeed = 0.05f; // 移动速度

    public GameObject leftRobotFrontBack;
    public GameObject leftRobotLeftRight;
    public GameObject leftRobotUpDown;
    public GameObject rightRobotFrontBack;
    public GameObject rightRobotLeftRight;
    public GameObject rightRobotUpDown;

    private Vector3 leftFrontBackMovement = Vector3.zero;
    private Vector3 leftLeftRightMovement = Vector3.zero;
    private Vector3 leftUpDownMovement = Vector3.zero;
    private Vector3 rightFrontBackMovement = Vector3.zero;
    private Vector3 rightLeftRightMovement = Vector3.zero;
    private Vector3 rightUpDownMovement = Vector3.zero;

    void Update()
    {
        // 左侧机器人移动
        leftUpDownMovement.y = (Input.GetKey(KeyCode.W) ? 1 : 0) + (Input.GetKey(KeyCode.S) ? -1 : 0);
        leftLeftRightMovement.x = (Input.GetKey(KeyCode.A) ? -1 : 0) + (Input.GetKey(KeyCode.D) ? 1 : 0);
        leftFrontBackMovement.z = (Input.GetKey(KeyCode.Z) ? -1 : 0) + (Input.GetKey(KeyCode.X) ? 1 : 0);

        // 右侧机器人移动
        rightUpDownMovement.y = (Input.GetKey(KeyCode.I) ? 1 : 0) + (Input.GetKey(KeyCode.K) ? -1 : 0);
        rightLeftRightMovement.x = (Input.GetKey(KeyCode.J) ? -1 : 0) + (Input.GetKey(KeyCode.L) ? 1 : 0);
        rightFrontBackMovement.z = (Input.GetKey(KeyCode.N) ? -1 : 0) + (Input.GetKey(KeyCode.M) ? 1 : 0);
        // 右侧机器人Z轴使用UI控制, 在外部UI事件中处理 rightFrontBackMovement

        // 移动物体
        if (leftRobotUpDown != null)
            leftRobotUpDown.transform.position += leftUpDownMovement.normalized * moveSpeed * Time.deltaTime;
        if (leftRobotLeftRight != null)
            leftRobotLeftRight.transform.position += leftLeftRightMovement.normalized * moveSpeed * Time.deltaTime;
        if (leftRobotFrontBack != null)
            leftRobotFrontBack.transform.position += leftFrontBackMovement.normalized * moveSpeed * Time.deltaTime;

        if (rightRobotUpDown != null)
            rightRobotUpDown.transform.position += rightUpDownMovement.normalized * moveSpeed * Time.deltaTime;
        if (rightRobotLeftRight != null)
            rightRobotLeftRight.transform.position += rightLeftRightMovement.normalized * moveSpeed * Time.deltaTime;
        if (rightRobotFrontBack != null)
            rightRobotFrontBack.transform.position += rightFrontBackMovement.normalized * moveSpeed * Time.deltaTime;
    }
}
