using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform target; 
    public float rotationSpeed = 100f; 
    private float yaw;
    private float pitch;
    private float radius; 
    private Vector3 initialOffset;
    private Quaternion initialRotation;

    void Start()
    {
        if (target == null)
        {
            Debug.LogError("CameraController: Target is not assigned!");
            return;
        }

        initialOffset = transform.position - target.position;
        radius = initialOffset.magnitude;
        initialRotation = transform.rotation;

        yaw = transform.eulerAngles.y;
        pitch = transform.eulerAngles.x;
    }

    void Update()
    {
        if (target == null) return;


        float horizontal = 0f;
        float vertical = 0f;

        if (Input.GetKey(KeyCode.LeftArrow)) horizontal = -1f;
        if (Input.GetKey(KeyCode.RightArrow)) horizontal = 1f;
        if (Input.GetKey(KeyCode.UpArrow)) vertical = 1f;
        if (Input.GetKey(KeyCode.DownArrow)) vertical = -1f;

        yaw += horizontal * rotationSpeed * Time.deltaTime;
        pitch += vertical * rotationSpeed * Time.deltaTime;
        pitch = Mathf.Clamp(pitch, -80f, 80f); 
        Quaternion rotation = initialRotation * Quaternion.Euler(pitch, yaw, 0f);
        Vector3 offset = rotation * initialOffset;
        transform.position = target.position + offset;

        transform.LookAt(target.position);
    }

}
