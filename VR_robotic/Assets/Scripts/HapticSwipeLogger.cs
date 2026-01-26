using System;
using System.Collections.Generic;
using UnityEngine;

public class HapticSwipeLogger : MonoBehaviour
{
    public HapticSwipe hapticSwipe; // 关联 HapticSwipe 组件
    private List<DateTime> leftSwipeTimes = new List<DateTime>();  // 记录左滑时间
    private List<DateTime> rightSwipeTimes = new List<DateTime>(); // 记录右滑时间

    void Start()
    {
        if (hapticSwipe == null)
        {
            Debug.LogError("HapticSwipeLogger: 未绑定 HapticSwipe 组件！");
            return;
        }

        // 监听 HapticSwipe 事件
        hapticSwipe.GestureEvents.OnSwipeLeft.AddListener(LogLeftSwipe);
        hapticSwipe.GestureEvents.OnSwipeRight.AddListener(LogRightSwipe);
    }

    void LogLeftSwipe()
    {
        DateTime currentTime = DateTime.Now;
        leftSwipeTimes.Add(currentTime);
        Debug.Log($"[HapticSwipeLogger] 左滑事件发生，时间：{currentTime}");
    }

    void LogRightSwipe()
    {
        DateTime currentTime = DateTime.Now;
        rightSwipeTimes.Add(currentTime);
        Debug.Log($"[HapticSwipeLogger] 右滑事件发生，时间：{currentTime}");
    }
}
