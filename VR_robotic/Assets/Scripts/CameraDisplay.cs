using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class CameraDisplay : MonoBehaviour
{
    public RawImage displayImage;         // UI上用于显示相机画面的 RawImage
    public RectTransform displayRect;     // RawImage 对应的 RectTransform
    public string cameraName = "USB camera";  // 你可以在 Inspector 中指定或更换名称

    private WebCamTexture webCamTexture;

    void Start()
    {
        if (WebCamTexture.devices.Length > 0)
        {
            // 创建 WebCamTexture，指定分辨率为 1600x1200
            webCamTexture = new WebCamTexture(cameraName, 1600, 1200);
            displayImage.texture = webCamTexture;
            displayImage.material.mainTexture = webCamTexture;
            webCamTexture.Play();

            StartCoroutine(AdjustAspect());
        }
        else
        {
            Debug.LogError("No camera detected.");
        }
        // 设置左上角显示
        if (displayRect != null)
        {
            displayRect.anchorMin = new Vector2(0f, 1f);    // 左上角锚点
            displayRect.anchorMax = new Vector2(0f, 1f);
            displayRect.pivot = new Vector2(0f, 1f);        // UI 相对于左上角展开
            displayRect.anchoredPosition = new Vector2(10, -10); // 向右下稍微偏移，避免贴边
        }

    }

    IEnumerator AdjustAspect()
    {
        // 等待直到摄像头准备就绪
        yield return new WaitUntil(() => webCamTexture.width > 100);

        // 保持相机图像比例
        float aspect = (float)webCamTexture.width / webCamTexture.height;  // 1600/1200 = 1.33
        float targetHeight = 600f;
        float targetWidth = targetHeight * aspect;

        displayRect.sizeDelta = new Vector2(targetWidth, targetHeight);

        // 可选：水平镜像图像（部分相机可能默认是翻转的）
        // displayRect.localScale = new Vector3(-1, 1, 1);
    }

    void OnDestroy()
    {
        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            webCamTexture.Stop();
        }
    }
}
