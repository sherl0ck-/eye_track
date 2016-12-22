//
//  VideoViewController.h
//  DataCollection
//
//  Created by Frederik Jensen on 30/10/16.
//  Copyright (c) 2016 Brinck10. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import "CustomView.h"

@interface VideoViewController : NSViewController {
    AVCaptureSession *session;
    AVCaptureConnection *videoConnection;
    AVCaptureStillImageOutput *stillImageOutput;
}

@property (nonatomic, retain) AVCaptureVideoPreviewLayer *preview;

@property (nonatomic, retain) CustomView *customView;
@property (nonatomic, retain) NSTimer *timer;
@property (nonatomic, retain) NSFileHandle *fileHandle;
@property (nonatomic, retain) NSString *path;
@property (nonatomic, retain) NSString *pointPath;

@property (nonatomic, assign) NSInteger width;
@property (nonatomic, assign) NSInteger height;
@property (nonatomic, assign) NSInteger posX;
@property (nonatomic, assign) NSInteger posY;
@property (nonatomic, assign) NSInteger oldY;
@property (nonatomic, assign) NSInteger speed;
@property (nonatomic, assign) BOOL direction;
@property (nonatomic, assign) BOOL moveX;


@property (nonatomic, retain) IBOutlet NSButton *startCollecting;
@property (nonatomic, retain) IBOutlet NSImageView *imageView;

- (IBAction) startCollection_click : (id) sender;

-(void) initCaptureSession;
-(void) setupPreviewLayer;

@end
