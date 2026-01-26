using System.Collections.Generic;
using UnityEngine;

public class ObjectMovement : MonoBehaviour
{
    public float speed = 5f; // 物体移动速度
    public GameObject targetObject; // 用户选择的物体
    public List<GameObject> followers = new List<GameObject>(); // 需要跟随的物体列表
    public int position = 0;//left=1,right=0

    void Update()
    {
        if (position == 1)
        {
            if (targetObject == null) return; // 确保有目标物体

            // 获取玩家输入 (-1 = Q, 1 = E)
            float moveInput = 0f;
            if (Input.GetKey(KeyCode.Q)) moveInput = -1f;
            if (Input.GetKey(KeyCode.E)) moveInput = 1f;

            // 计算目标物体的移动向量（沿局部 X 轴）
            Vector3 moveVector = targetObject.transform.right * moveInput * speed * Time.deltaTime;

            // 移动目标物体
            targetObject.transform.position += moveVector;

            // 让 followers 列表中的所有物体沿相同的向量方向移动
            foreach (GameObject follower in followers)
            {
                if (follower != null) // 确保物体存在
                {
                    follower.transform.position += moveVector;
                }
            }
        }
        else
        {
            if (targetObject == null) return; // 确保有目标物体

            // 获取玩家输入 (-1 = Q, 1 = E)
            float moveInput = 0f;
            if (Input.GetKey(KeyCode.U)) moveInput = -1f;
            if (Input.GetKey(KeyCode.O)) moveInput = 1f;

            // 计算目标物体的移动向量（沿局部 X 轴）
            Vector3 moveVector = targetObject.transform.right * moveInput * speed * Time.deltaTime;

            // 移动目标物体
            targetObject.transform.position += moveVector;

            // 让 followers 列表中的所有物体沿相同的向量方向移动
            foreach (GameObject follower in followers)
            {
                if (follower != null) // 确保物体存在
                {
                    follower.transform.position += moveVector;
                }
            }
        }
    }
}
