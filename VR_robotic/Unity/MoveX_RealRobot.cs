using UnityEngine;

public class MoveX_RealRobot : MonoBehaviour
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

        // Set the new position, updating only the X coordinate
        transform.position = new Vector3(newX-0.1f,startPosition.y, newY + 1.058f);
        
    }
}