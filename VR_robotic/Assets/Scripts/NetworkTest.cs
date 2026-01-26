using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class NetworkTest : MonoBehaviour
{
    void Start()
    {
        StartCoroutine(TestConnection());
    }

    IEnumerator TestConnection()
    {
        string testUrl = "https://wit.ai/apps/637521292734063";
        using (UnityWebRequest request = UnityWebRequest.Get(testUrl))
        {
            yield return request.SendWebRequest();

#if UNITY_2020_1_OR_NEWER
            if (request.result == UnityWebRequest.Result.Success)
#else
            if (!request.isNetworkError && !request.isHttpError)
#endif
            {
                Debug.Log("Network test success: " + request.downloadHandler.text.Substring(0, 50) + "...");
            }
            else
            {
                Debug.LogError("Network test failed: " + request.error);
            }
        }
    }
}
