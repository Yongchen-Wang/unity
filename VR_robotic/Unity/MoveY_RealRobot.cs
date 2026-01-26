using UnityEngine;

public class MoveY_RealRobot : MonoBehaviour
{
    public float positionOffset = 10000; // Position offset for Y axis
    public float scale = 10000 * 40; // Scaling factor for Y axis
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
    }

    void Update()
    {
        // Get the latest Y position from the UDP Receiver and apply offset and scaling
        float newY = -(TCPReceiver.latestPosition.y - positionOffset) / scale;

        // Set the new position, updating only the Y coordinate
        transform.position = new Vector3(startPosition.x, startPosition.y, newY+ 1.058f);//
    }
}
