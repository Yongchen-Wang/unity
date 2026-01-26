using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Threading;


public class BallPositionReceiver : MonoBehaviour
{
    public float positionOffset = 10000; // Position offset for all axes
    public float scale = 100000; // Scaling factor for all axes
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.localPosition; // Use localPosition for local coordinate space
    }

    void Update()
    {
        // Get the latest position from the TCP Receiver and apply offset and scaling

        float newX = -TCPReceiver.latestPosition.x / scale;
        float newY = 0.03f;
        float newZ = TCPReceiver.latestPosition.y / scale;





        // Set the new local position, updating all coordinates
        transform.localPosition = new Vector3(newX, newY, newZ);
    }
}
