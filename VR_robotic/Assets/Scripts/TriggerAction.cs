using UnityEngine;

public class TriggerAction : MonoBehaviour
{
    // 当另一个 Collider 持续处于触发区域内时，每帧调用
    void OnCollisionStay(Collision collision)
    {

            DoSomething();

    }


    void DoSomething()
    {
        Debug.Log("两个物体正在相交，持续执行此函数");
        // 在这里放你想执行的逻辑
    }
}
