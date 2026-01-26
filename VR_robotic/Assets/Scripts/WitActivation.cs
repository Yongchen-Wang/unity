using Oculus.Voice;
using UnityEngine;

public class WitActivation : MonoBehaviour
{
    private AppVoiceExperience _voiceExperience;
    private void OnValidate()
    {
        if (!_voiceExperience) _voiceExperience = GetComponent<AppVoiceExperience>();
    }

    private void Start()
    {
        //string deviceName = Microphone.devices[1]; // 选择第一个麦克风设备
        //Debug.Log(deviceName);
        //int minFreq = 44100; // 最小频率
        //AudioClip audioClip = Microphone.Start(deviceName, true, 10, minFreq);
        _voiceExperience = GetComponent<AppVoiceExperience>();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("*** Pressed Space bar ***");
            ActivateWit();
        }
    }

    /// <summary>
    /// Activates Wit i.e. start listening to the user.
    /// </summary>
    public void ActivateWit()
    {
        _voiceExperience.Activate();
    }
}