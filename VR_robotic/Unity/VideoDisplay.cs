
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VideoDisplay : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        // Create a WebCamTexture
        WebCamTexture webCamTexture = new WebCamTexture();

        // Assign the texture to the renderer's material
        GetComponent<Renderer>().material.mainTexture = webCamTexture;

        // Start the WebCamTexture
        webCamTexture.Play();
    }
}


/*
using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.IO;
using System;

public class VideoDisplay : MonoBehaviour
{
    public string serverIP = "YOUR_SERVER_IP";
    public int serverPort = 9999;
    private TcpClient client;
    private NetworkStream stream;
    private byte[] imageData;
    private bool stop = false;

    void Start()
    {
        client = new TcpClient();
        client.Connect(serverIP, serverPort);
        stream = client.GetStream();
        StartCoroutine(ReceiveData());
    }

    IEnumerator ReceiveData()
    {
        byte[] imageSizeBytes = new byte[8];
        while (!stop)
        {
            if (stream.Read(imageSizeBytes, 0, imageSizeBytes.Length) != imageSizeBytes.Length)
            {
                stop = true;
                yield break;
            }

            int imageSize = BitConverter.ToInt32(imageSizeBytes, 0);
            imageData = new byte[imageSize];
            int totalRead = 0;
            while (totalRead < imageSize)
            {
                int read = stream.Read(imageData, totalRead, imageSize - totalRead);
                totalRead += read;
            }

            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(imageData);
            GetComponent<Renderer>().material.mainTexture = texture;
            yield return null;
        }
    }

    void OnApplicationQuit()
    {
        stop = true;
        stream.Close();
        client.Close();
    }
}
*/