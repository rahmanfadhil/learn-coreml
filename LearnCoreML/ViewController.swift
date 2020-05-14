//
//  ViewController.swift
//  LearnCoreML
//
//  Created by Rahman Fadhil on 14/05/20.
//  Copyright © 2020 Rahman Fadhil. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {

    @IBOutlet weak var takenImage: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    lazy var analyseRequest: VNCoreMLRequest = {
        do {
            // Initiate machine learning model
            let model = try VNCoreMLModel(for: LearnImageClassifier_1().model)
            let request = VNCoreMLRequest(model: model) { [weak self] (request, error) in
                // Ask the machine to evaluate the object and give the result based on our model
                self?.processToAnalyse(for: request, error: error)
            }
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load machine learning model: \(error)")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    @IBAction func takePicture(_ sender: UIButton) {
        openCameraAndLibrary()
    }
    
}

// MARK: Function to open image library and camera
extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    private func openCameraAndLibrary() {
        let imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        
        let actionAlert = UIAlertController(title: "Browse image", message: "Chose image source", preferredStyle: .actionSheet)
        
        // Give users option to take a picture with camera
        actionAlert.addAction(UIAlertAction(title: "Camera", style: .default, handler: { (action) in
            // Check if camera is accessible
            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                imagePicker.sourceType = .camera
                self.present(imagePicker, animated: true)
            } else {
                print("Camera is not available!")
            }
        }))
        
        // Give users option to pick an image from their library
        actionAlert.addAction(UIAlertAction(title: "Photo Library", style: .default, handler: { (action) in
            imagePicker.sourceType = .photoLibrary
            imagePicker.mediaTypes = ["public.image"]
            self.present(imagePicker, animated: true)
        }))
        
        // Cancel operation
        actionAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(actionAlert, animated: true)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let imageTaken = info[.originalImage] as? UIImage {
            picker.dismiss(animated: true) {
                self.takenImage.image = imageTaken
                // Convert image that to be supported by Vision Framework
                self.convertImageToAnalysed(image: imageTaken)
            }
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
}

// MARK: Machine learning process goes here
extension ViewController {
    private func convertImageToAnalysed(image: UIImage) {
        resultLabel.text = "Analysing..."
        
        // Converting image to a format that the Vision Framework support
        let imageProperty = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create \(CIImage.self) from \(image)")
        }
        
        // Create a background task to process the given image to avoid memory leak
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: imageProperty!)
            do {
                try handler.perform([self.analyseRequest])
            } catch {
                fatalError("Can perform image processing!")
            }
        }
    }

    private func processToAnalyse(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.resultLabel.text = "Can't analyse the object!"
                return
            }
            
            let classifications = results as! [VNClassificationObservation]
            if classifications.isEmpty {
                self.resultLabel.text = "Nothing to analysed!"
            } else {
                // Get only 2 top information from classification data, which contains confidence level and identifier value.
                let importantInformation = classifications.prefix(2)
                
                // Convert the top 2 information into a readable text
                let readableStringResult = importantInformation.map { (classification) in
                    return String(format: "(%.2f), %@", classification.confidence, classification.identifier)
                }
                self.resultLabel.text = readableStringResult.joined(separator: " | ")
            }
        }
    }
}
