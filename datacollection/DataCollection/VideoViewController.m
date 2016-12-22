//
//  VideoViewController.m
//  DataCollection
//
//  Created by Frederik Jensen on 30/10/16.
//  Copyright (c) 2016 Brinck10. All rights reserved.
//

#import "VideoViewController.h"

@implementation VideoViewController
@synthesize customView;

-(void) viewWillAppear {
    
    [super viewWillAppear];
  
    // Base configuration
    // #############################
    // ## MAKE SURE THIS PATH EXISTS
    // #############################
    self.path = @"/Users/Brinck/Documents/data/";

    
    // Setup movement controls
    self.width = 40;
    self.height = 40,
    self.posX = 0;
    self.posY = self.view.frame.size.height - self.height;
    self.speed = 20;
    self.direction = true;
    self.oldY = 0;
    self.moveX = true;
    
    // Paths
    self.pointPath = [NSString stringWithFormat:@"%@points", self.path];
    
    // Add custom view for graphics
    self.customView = [[CustomView alloc] initWithFrame:NSRectFromCGRect(CGRectMake(0.0F, self.view.frame.size.height - self.height, self.view.frame.size.width, self.view.frame.size.height))];
    [self.view setWantsLayer: YES];
    [self.customView setDelegate:self];
    [self.view addSubview: self.customView];

    // Setup capture
    [self initCaptureSession];
    [self setupPreviewLayer];
}


- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Do view setup here.
    
}

- (void)viewDidLayout {
    [super viewDidLayout];
    
    // Reposition graphics on window resizing
    self.customView.frame = NSRectFromCGRect(CGRectMake(0.0F, self.view.frame.size.height - self.height, self.view.frame.size.width, self.view.frame.size.height));
    self.posY = self.view.frame.size.height - self.height;
}


-(IBAction)startCollection_click:(id)sender {
    // Toggle data collection
    if(![session isRunning]) {
        self.fileHandle = [NSFileHandle fileHandleForWritingAtPath:self.pointPath];
        if (self.fileHandle == nil) {
            [[NSFileManager defaultManager] createFileAtPath:self.pointPath contents:nil attributes:nil];
            self.fileHandle = [NSFileHandle fileHandleForWritingAtPath:self.pointPath];
        }
        [self.fileHandle seekToEndOfFile];
        
        [session startRunning];
        self.timer = [NSTimer scheduledTimerWithTimeInterval:0.05 target:self selector:@selector(streamPictures:) userInfo:nil repeats:YES];
        [self.timer fire];
        self.startCollecting.hidden = true;
    }
}


-(void)initCaptureSession {
    // Start capturing session for
    // graphics capturing
    session = [[AVCaptureSession alloc] init];
    
    if ([session canSetSessionPreset:AVCaptureSessionPresetHigh])
        [session setSessionPreset:AVCaptureSessionPresetHigh];
    
    AVCaptureDeviceInput *input = [[AVCaptureDeviceInput alloc] initWithDevice:[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo][0] error : nil];
    if ([session canAddInput: input])
        [session addInput: input];
}


-(void) setupPreviewLayer {
    // Configure output settings
    self.preview = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];

    stillImageOutput = [[AVCaptureStillImageOutput alloc] init];
    NSDictionary *outputSettings =  [[NSDictionary alloc] initWithObjectsAndKeys:AVVideoCodecJPEG, AVVideoCodecKey, nil];
    [stillImageOutput setOutputSettings:outputSettings];
    if ([session canAddOutput:stillImageOutput])
        [session addOutput:stillImageOutput];
}


-(void) streamPictures:(id) sender {
    // Get video port
    videoConnection = nil;
    for (AVCaptureConnection *connection in stillImageOutput.connections) {
        for (AVCaptureInputPort *port in [connection inputPorts])
            if([[port mediaType] isEqual:AVMediaTypeVideo]) {
                videoConnection = connection;
                break;
            }
        
        if(videoConnection)
            break;
    }
    
    // Capture image ouput
    [stillImageOutput captureStillImageAsynchronouslyFromConnection:videoConnection completionHandler:^(CMSampleBufferRef imageDataSampleBuffer, NSError *error) {
        if(imageDataSampleBuffer != nil) {
            // Turn image into savable png
            NSData *imageData = [AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageDataSampleBuffer];
            NSBitmapImageRep *representation = [[NSBitmapImageRep alloc] initWithData:imageData];
            NSData *pngData = [representation representationUsingType:NSPNGFileType properties:nil];
            
            // Get image name
            NSDateFormatter *dateFormatter=[[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd-HH-mm-ss-SSS"];
            NSString *name = [dateFormatter stringFromDate:[NSDate date]];
            NSString *path = [NSString stringWithFormat:@"%@c-%@.png", self.path, name];
            
            [self.fileHandle writeData:[[NSString stringWithFormat:@"c-%@.png, %ld, %ld\n", name, self.posX+self.width/2, self.posY-self.height/2] dataUsingEncoding:NSUTF8StringEncoding]];
            [pngData writeToFile:path atomically:YES];
        }
    }];
    
    [self.customView setFrame:NSMakeRect(self.posX, self.posY, self.width,self.height)];
    [self.customView updateLayer];
    
    // Update horizontal direction.
    if (self.posX > self.view.frame.size.width - self.width || self.posX < 0) {
        self.direction = !self.direction;
        self.moveX = false;
        self.posX = (self.posX < 0) ? 0 : self.view.frame.size.width - self.width;
    }
    
    // Check if we are moving horizontally; if not, move
    // vertically one unit of the square's height.
    if (self.moveX) {
        self.oldY = self.posY;
        self.posX = (self.direction) ? self.posX + self.speed : self.posX - self.speed;
    } else {
        self.posY -= self.speed;
        
        // We have moved enough vertically,
        // now move horizontally.
        if (self.posY >= self.oldY - self.height){
            self.posY = self.oldY - self.height;
            self.moveX = true;
        }
    }
    
    // Check for data collection session end
    // and allow for experiment restart
    if (self.oldY <= 0 - self.height) {
        [session stopRunning];
        [self.timer invalidate];
        [self.fileHandle closeFile];
        [self.startCollecting setHidden:false];
        self.customView.frame = NSRectFromCGRect(CGRectMake(0.0F, self.view.frame.size.height - self.height, self.view.frame.size.width, self.view.frame.size.height));
    }
        
}
@end
