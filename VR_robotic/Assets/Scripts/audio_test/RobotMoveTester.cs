using UnityEngine;

public class RobotMoveTester : MonoBehaviour
{
    /// <summary>
    /// 模拟语音输出调试，输入三个字符串：robot, direction, distance
    /// </summary>
    /// <param name="values">[0] = robot, [1] = direction, [2] = distance</param>
    public void TestDebugOutput(string[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            Debug.Log(values[i]);
        }
        //if (values.Length < 3)
        //{
        //    Debug.LogWarning("输入参数不足，需要 [robot, direction, distance]");
        //    return;
        //}

        //string robot = values[0];
        //string direction = values[1];
        //string distance = values[2];

        //Debug.Log($"[Voice → Debug] Robot hand = {robot}");
        //Debug.Log($"[Voice → Debug] Direction = {direction}");
        //Debug.Log($"[Voice → Debug] Distance = {distance}");
    }
}