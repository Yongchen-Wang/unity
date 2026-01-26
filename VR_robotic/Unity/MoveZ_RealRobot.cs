using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveZ_RealRobot : MonoBehaviour
{
    public float positionOffset = 10000; // Position offset
    public float scale = 10000 * 40; // Scaling factor
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
    }

    void Update()
    {
        // Get the latest X position from the UDP Receiver and apply offset and scaling
        float newX = (TCPReceiver.latestPosition.x - positionOffset) / scale;
        float newY = -(TCPReceiver.latestPosition.y - positionOffset) / scale;
        float newZ = -(TCPReceiver.latestPosition.z - positionOffset) / scale;

        // Set the new position, updating only the X coordinate
        transform.position = new Vector3(newX-0.02f, newZ+ 1.245f, newY + 1.058f);//1.225f

    }
}